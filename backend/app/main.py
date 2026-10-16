"""
Phases 14-15 - FastAPI backend.   Run: uvicorn backend.app.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import os
import re
import json
import datetime
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler

from backend.app import db
from backend.app.security import verify_signature

load_dotenv()
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class RepoIn(BaseModel):
    full_name: str


def run_analysis(full_name: str):
    from backend.app.pipeline import analyze
    try:
        result = analyze(full_name)
        db.repos.update_one({"full_name": full_name}, {"$set": result}, upsert=True)
    except Exception as e:  # keep the API alive, show the error to the user
        db.repos.update_one({"full_name": full_name},
                            {"$set": {"status": "error", "error": str(e)[:300]}}, upsert=True)


def refresh_job():
    cutoff = (datetime.datetime.utcnow() - datetime.timedelta(days=31)).isoformat()
    for doc in db.repos.find({"$or": [{"needs_refresh": True}, {"updated_at": {"$lt": cutoff}}]},
                             {"full_name": 1}):
        run_analysis(doc["full_name"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        db.init_indexes()
    except Exception as e:
        print(f"WARNING: MongoDB not reachable ({e}). Start the MongoDB service.")
    sched = BackgroundScheduler()
    sched.add_job(refresh_job, "interval", hours=24)
    sched.start()
    yield
    sched.shutdown()


app = FastAPI(title="OSS Health Predictor", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"],
                   allow_methods=["*"], allow_headers=["*"])


def get_doc(owner, name):
    doc = db.repos.find_one({"full_name": f"{owner}/{name}"}, {"_id": 0})
    if not doc:
        raise HTTPException(404, "Repository not tracked yet. POST /repos first.")
    return doc


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/repos")
def add_repo(body: RepoIn, bg: BackgroundTasks):
    name = body.full_name.strip()
    if not NAME_RE.match(name):
        raise HTTPException(422, "Use the format owner/repo, e.g. psf/requests")
    db.repos.update_one({"full_name": name},
                        {"$set": {"full_name": name, "status": "processing", "error": None}}, upsert=True)
    bg.add_task(run_analysis, name)
    return {"full_name": name, "status": "processing"}


@app.get("/repos")
def list_repos():
    return list(db.repos.find({}, {"_id": 0, "full_name": 1, "status": 1, "prediction": 1,
                                   "as_of_month": 1, "probabilities": 1}))


@app.get("/repos/{owner}/{name}")
def repo_detail(owner: str, name: str):
    return get_doc(owner, name)

@app.delete("/repos/{owner}/{name}")
def delete_repo(owner: str, name: str):
    full = f"{owner}/{name}"
    if db.repos.delete_one({"full_name": full}).deleted_count == 0:
        raise HTTPException(404, "Repository not tracked.")
    db.snapshots.delete_many({"repository": full.replace("/", "_")})
    db.counters.delete_many({"repo": full})
    return {"deleted": full}


@app.get("/repos/{owner}/{name}/health")
def repo_health(owner: str, name: str):
    d = get_doc(owner, name)
    return {k: d.get(k) for k in ["full_name", "status", "error", "prediction", "model_prediction",
                                  "already_inactive", "probabilities", "as_of_month", "horizon", "updated_at"]}


@app.get("/repos/{owner}/{name}/history")
def repo_history(owner: str, name: str):
    return get_doc(owner, name).get("history", [])


@app.get("/repos/{owner}/{name}/explain")
def repo_explain(owner: str, name: str):
    return get_doc(owner, name).get("top_factors", [])


@app.get("/repos/{owner}/{name}/recommendations")
def repo_recs(owner: str, name: str):
    return get_doc(owner, name).get("recommendations", [])


@app.post("/repos/{owner}/{name}/refresh")
def refresh(owner: str, name: str, bg: BackgroundTasks):
    get_doc(owner, name)
    db.repos.update_one({"full_name": f"{owner}/{name}"}, {"$set": {"status": "processing"}})
    bg.add_task(run_analysis, f"{owner}/{name}")
    return {"status": "processing"}


@app.get("/models/info")
def models_info():
    from backend.app.pipeline import model_info
    try:
        return model_info()
    except FileNotFoundError:
        raise HTTPException(404, "No trained model found. Run ml.models.train_models first.")


@app.post("/webhooks/github")
async def github_webhook(request: Request):
    body = await request.body()
    secret = os.getenv("GITHUB_WEBHOOK_SECRET", "")
    if not verify_signature(secret, body, request.headers.get("X-Hub-Signature-256", "")):
        raise HTTPException(401, "Invalid signature")
    event = request.headers.get("X-GitHub-Event", "")
    p = json.loads(body)
    repo = (p.get("repository") or {}).get("full_name")
    if not repo:
        return {"ok": True, "ignored": True}

    inc = {}
    if event == "push":
        inc["commits"] = len(p.get("commits", []))
    elif event == "pull_request":
        if p.get("action") == "opened":
            inc["prs_opened"] = 1
        elif p.get("action") == "closed" and p["pull_request"].get("merged"):
            inc["prs_merged"] = 1
    elif event == "issues":
        if p.get("action") == "opened":
            inc["issues_opened"] = 1
        elif p.get("action") == "closed":
            inc["issues_closed"] = 1
    if inc:
        ym = datetime.datetime.utcnow().strftime("%Y-%m")
        db.counters.update_one({"key": f"{repo}|{ym}"},
                               {"$inc": inc, "$set": {"repo": repo, "year_month": ym}}, upsert=True)
        db.repos.update_one({"full_name": repo}, {"$set": {"needs_refresh": True}})
    return {"ok": True, "event": event}
