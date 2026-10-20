"""Optional, attributed work notes. These never count as local picture evidence."""

import json
import re
import time
import math
import requests

from app.services.library.store import stable_hash


def save_work(store, library_id, work):
    from app.services.library.runner import library_operation
    with library_operation(library_id):
        previous = store.cache_get(library_id, "work") or {}
        value = {k: work.get(k, previous.get(k, default)) for k, default in
                 (("title", ""), ("adaptation", "动画"), ("season", ""), ("notes", ""),
                  ("characters", []), ("external", {}), ("episode_map", {}))}
        for field in ("title","adaptation","season","notes"):
            if not isinstance(value[field],str):
                raise ValueError(f"{field}须为文本。")
        if not isinstance(value["characters"],list):
            raise ValueError("人物卡应为列表。")
        for person in value["characters"]:
            if not isinstance(person,dict) or not isinstance(person.get("name"),str) or not person["name"].strip():
                raise ValueError("每张人物卡需要名称。")
            for field in ("aliases","reference_ranges"):
                items = person.get(field,[])
                if not isinstance(items,list) or any(not isinstance(i,str) for i in items):
                    raise ValueError(f"人物卡{field}应为文本列表。")
        if not isinstance(value["episode_map"],dict) or any(not str(k).isdigit() or type(v) not in (int,float) or
                not math.isfinite(v) or v <= 0 for k,v in value["episode_map"].items()):
            raise ValueError("集号映射须为有效正数编号。")
        value["revision"] = stable_hash(value)
        if value.get("revision") == previous.get("revision"):
            return previous
        if previous:
            store.cache_put(library_id, "work-history:"+previous["revision"], previous)
        value["updated"] = time.time()
        store.cache_put(library_id, "work", value)
        return value


def fetch_bangumi(subject, proxy=None):
    """Only a public subject ID is sent; media, scripts and keys stay out of this request."""
    match = re.fullmatch(r"(?:https://(?:bgm\.tv|bangumi\.tv|chii\.in)/subject/)?([0-9]+)", str(subject).strip())
    if not match:
        raise ValueError("请输入 Bangumi 作品编号或 https://bgm.tv/subject/编号。")
    subject_id = int(match[1])
    proxies = None
    if proxy and proxy.get("enabled") is False:
        proxies = {"http": None, "https": None}
    elif proxy and (proxy.get("http") or proxy.get("https")):
        proxies = {"http": proxy.get("http") or proxy.get("https"), "https": proxy.get("https") or proxy.get("http")}

    def get(path, **params):
        response = requests.get("https://api.bgm.tv/v0/"+path, params=params,
                                headers={"User-Agent": "NarraLens/2.8.0 (https://github.com/SylphyZ/NarraLens)"},
                                proxies=proxies, timeout=(10, 30))
        response.raise_for_status()
        return response.json()

    subject_data = get(f"subjects/{subject_id}")
    characters = get(f"subjects/{subject_id}/characters")
    episodes, offset = [], 0
    while True:
        page = get("episodes", subject_id=subject_id, type=0, limit=100, offset=offset)
        rows = page.get("data", [])
        episodes.extend(rows)
        offset += len(rows)
        if not rows or offset >= page.get("total", offset):
            break
    return {"source": f"https://bgm.tv/subject/{subject_id}", "fetched_at": time.time(),
            "provider": "Bangumi", "source_type": subject_data.get("type"),
            "usage": "公开资料，仅辅助识别；保留出处，不作为本地画面证据。",
            "title": subject_data.get("name_cn") or subject_data.get("name", ""),
            "summary": subject_data.get("summary", ""),
            "characters": [{"name": c.get("name", ""), "role": c.get("relation", ""),
                            "source": f"https://bgm.tv/character/{c['id']}",
                            "image": (c.get("images") or {}).get("medium", "")} for c in characters],
            "episodes": [{"number": e.get("sort"), "title": e.get("name_cn") or e.get("name"),
                          "summary": e.get("desc", ""), "source": f"https://bgm.tv/ep/{e['id']}"} for e in episodes]}


def episode_context(work, episode):
    """Small relevant cards; large user notes remain local and are selected by heading."""
    mapped = work.get("episode_map", {}).get(str(episode), episode)
    external = work.get("external", {})
    summaries = [e for e in external.get("episodes", []) if e.get("number") == mapped]
    # Unscoped notes are global; headings such as '## 第 4 集' scope long notes.
    notes = []
    active = True
    for line in str(work.get("notes", "")).splitlines():
        heading = re.match(r"^#{1,6}\s*第\s*(\d+)\s*[集话話]", line)
        if heading:
            active = int(heading[1]) == int(episode)
        if active:
            notes.append(line)
    characters = work.get("characters") or external.get("characters", [])
    return {"work": work.get("title", ""), "adaptation": work.get("adaptation", ""),
            "season": work.get("season", ""), "episode": episode, "mapped_episode": mapped,
            "episode_notes": summaries, "characters": characters,
            "notes": "\n".join(notes), "external_is_context_only": True}


def context_digest(work, episode):
    return stable_hash(episode_context(work, episode))
