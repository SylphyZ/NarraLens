"""Versioned SQLite storage. No source-video copy or destructive migration."""

import hashlib
import json
import sqlite3
import time
import os
import shutil
import threading
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID, uuid4


from app.models.library import FRAME_PROTOCOL, LIBRARY_SCHEMA

SCHEMA_VERSION = LIBRARY_SCHEMA
_json_lock_guard = threading.Lock()
_json_locks = {}
_JSON_MISSING = object()


@contextmanager
def _json_access(path):
    """Readers and replacers in this process must not hold Windows handles together."""
    key = os.path.normcase(os.path.abspath(os.fspath(path)))
    with _json_lock_guard:
        entry = _json_locks.get(key)
        if entry is None:
            entry = [threading.RLock(), 0]
            _json_locks[key] = entry
        entry[1] += 1
    try:
        with entry[0]:
            yield
    finally:
        with _json_lock_guard:
            entry[1] -= 1
            if not entry[1] and _json_locks.get(key) is entry:
                del _json_locks[key]


def _retry_json_io(operation):
    delays = (.025, .05, .1, .2, .4, .8)
    for attempt in range(len(delays) + 1):
        try:
            return operation()
        except OSError as exc:
            if os.name != "nt" or getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == len(delays):
                raise
            time.sleep(delays[attempt])


def read_json(path, default=_JSON_MISSING):
    path = Path(path)
    with _json_access(path):
        try:
            return _retry_json_io(lambda: json.loads(path.read_text(encoding="utf-8-sig")))
        except FileNotFoundError:
            if default is _JSON_MISSING:
                raise
            return default


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def library_ids(value):
    """Read a selection or task snapshot, including original single-library tasks."""
    if isinstance(value, dict):
        value = value.get("library_ids") or value.get("library_id")
    if value is None:
        return []
    if isinstance(value, (str, UUID)):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise ValueError("素材库选择格式不正确。")
    return list(dict.fromkeys(str(UUID(str(identifier))) for identifier in value))


def write_json(path, value, *, compact=False, create_parent=True):
    path = Path(path)
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    serialized = json.dumps(value, ensure_ascii=False, indent=None if compact else 2,
                            separators=(",", ":") if compact else None)
    with _json_access(path):
        try:
            _retry_json_io(lambda: temporary.write_text(serialized, encoding="utf-8"))
            _retry_json_io(lambda: temporary.replace(path))
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                # An external process can also briefly hold our temporary file.
                pass


class LibraryStore:
    def __init__(self, root=None):
        self.root = Path(root) if root else Path(__file__).resolve().parents[3] / "storage" / "libraries"
        self.root.mkdir(parents=True, exist_ok=True)
        with self.connection(self.root / "catalog.sqlite") as db:
            db.execute("CREATE TABLE IF NOT EXISTS libraries (id TEXT PRIMARY KEY, name TEXT NOT NULL, settings TEXT NOT NULL DEFAULT '{}', created REAL NOT NULL)")

    @contextmanager
    def connection(self, database):
        db = sqlite3.connect(str(database), timeout=30)
        db.row_factory = sqlite3.Row
        try:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, SCHEMA_VERSION):
                raise ValueError("素材库版本与当前代码不兼容，请恢复对应版本的代码或数据库备份。")
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA journal_mode=WAL")
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def list_libraries(self):
        with self.connection(self.root / "catalog.sqlite") as db:
            rows = db.execute("SELECT * FROM libraries ORDER BY created DESC").fetchall()
        return [{**dict(row), "settings": json.loads(row["settings"])} for row in rows]

    def directory(self, library_id):
        library_id = str(UUID(str(library_id)))
        if not any(item["id"] == library_id for item in self.list_libraries()):
            raise ValueError("素材库不存在。")
        directory = self.root / library_id
        if directory.is_symlink() or directory.is_junction() or directory.resolve().parent != self.root.resolve():
            raise ValueError("素材库目录不在预期范围内或是链接，拒绝访问。")
        return directory

    @contextmanager
    def database(self, library_id):
        with self.connection(self.directory(library_id) / "library.sqlite") as db:
            yield db

    def create(self, name):
        name = str(name).strip()
        if not name:
            raise ValueError("请输入素材库名称。")
        identifier = str(uuid4())
        directory = self.root / identifier
        directory.mkdir()
        with self.connection(directory / "library.sqlite") as db:
            db.executescript("""
                CREATE TABLE assets (
                    id TEXT PRIMARY KEY, path TEXT NOT NULL UNIQUE, episode INTEGER NOT NULL,
                    size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, fingerprint TEXT NOT NULL,
                    duration REAL NOT NULL, width INTEGER, height INTEGER, fps REAL,
                    active_subtitle TEXT, status TEXT NOT NULL DEFAULT 'pending'
                );
                CREATE TABLE subtitle_sources (
                    id TEXT PRIMARY KEY, asset_id TEXT NOT NULL REFERENCES assets(id),
                    source_path TEXT NOT NULL, original_path TEXT NOT NULL,
                    processed_path TEXT NOT NULL, content_hash TEXT NOT NULL, settings TEXT NOT NULL
                );
                CREATE TABLE scenes (
                    id TEXT PRIMARY KEY, asset_id TEXT NOT NULL REFERENCES assets(id),
                    start REAL NOT NULL, end REAL NOT NULL, transcript TEXT NOT NULL,
                    analysis TEXT NOT NULL, version TEXT NOT NULL, visual TEXT NOT NULL DEFAULT '{}',
                    frame_paths TEXT NOT NULL DEFAULT '[]'
                );
                CREATE INDEX scenes_asset_time ON scenes(asset_id, start);
                CREATE TABLE shots (
                    id TEXT PRIMARY KEY, scene_id TEXT NOT NULL REFERENCES scenes(id),
                    start REAL NOT NULL, end REAL NOT NULL, description TEXT NOT NULL, version TEXT NOT NULL
                );
                CREATE TABLE analysis_jobs (
                    id TEXT PRIMARY KEY, state TEXT NOT NULL, progress REAL NOT NULL,
                    message TEXT NOT NULL, updated REAL NOT NULL, settings TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE analysis_cache (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE shot_records (
                    id TEXT PRIMARY KEY, asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
                    start REAL NOT NULL, end REAL NOT NULL, data TEXT NOT NULL
                );
                CREATE INDEX shot_records_asset ON shot_records(asset_id,start);
                CREATE TABLE content_segments (
                    id TEXT PRIMARY KEY, asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
                    start REAL NOT NULL, end REAL NOT NULL, data TEXT NOT NULL
                );
                CREATE TABLE content_links (
                    segment_id TEXT NOT NULL REFERENCES content_segments(id) ON DELETE CASCADE,
                    range_id TEXT NOT NULL REFERENCES shot_records(id) ON DELETE CASCADE,
                    PRIMARY KEY(segment_id,range_id)
                );
                PRAGMA user_version=2;
            """)
        with self.connection(self.root / "catalog.sqlite") as db:
            db.execute("INSERT INTO libraries(id,name,created) VALUES(?,?,?)", (identifier, name, time.time()))
        return identifier

    def save_settings(self, library_id, settings):
        self.directory(library_id)
        with self.connection(self.root / "catalog.sqlite") as db:
            db.execute("UPDATE libraries SET settings=? WHERE id=?", (json.dumps(settings, ensure_ascii=False), library_id))

    def schema(self, library_id):
        with self.database(library_id) as db:
            return db.execute("PRAGMA user_version").fetchone()[0]

    def require_content_library(self, identifiers):
        outdated = [i for i in library_ids(identifiers) if self.schema(i) != SCHEMA_VERSION]
        if outdated:
            raise ValueError("选中了旧窗口库。请在视频素材库中创建“2.8 拉片副本”并完成拉片，再在制作工作台选择新库；原配音、字幕和旧库保留。")

    def clone_for_analysis(self, library_id, name):
        """Copy registrations and owned subtitle files into a separate database."""
        from app.services.library.runner import library_operation
        with library_operation(library_id):
            identifier = self.create(name)
            target = self.directory(identifier)
            with self.database(library_id) as source, self.database(identifier) as dest:
                assets = [dict(row) for row in source.execute("SELECT * FROM assets")]
                for asset in assets:
                    asset["status"] = "pending"
                    dest.execute("INSERT INTO assets VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                                 tuple(asset[k] for k in ("id","path","episode","size","mtime_ns","fingerprint","duration","width","height","fps","active_subtitle","status")))
                for row in source.execute("SELECT * FROM subtitle_sources"):
                    data = dict(row)
                    for field in ("original_path", "processed_path"):
                        original = Path(data[field])
                        destination = target/"subtitles"/data["asset_id"]/data["id"]/original.name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(original, destination)
                        data[field] = str(destination)
                    dest.execute("INSERT INTO subtitle_sources VALUES(?,?,?,?,?,?,?)",
                                 tuple(data[k] for k in ("id","asset_id","source_path","original_path","processed_path","content_hash","settings")))
            settings = next(r["settings"] for r in self.list_libraries() if r["id"] == library_id)
            self.save_settings(identifier, {**settings, "source_library_id": library_id,
                                           "schema": SCHEMA_VERSION, "cloned_at": time.time()})
            return identifier

    def save_ranges(self, library_id, asset_id, ranges):
        self.require_content_library(library_id)
        with self.database(library_id) as db:
            self._write_ranges(db, asset_id, ranges)
            db.execute("UPDATE assets SET status='partial' WHERE id=?", (asset_id,))

    @staticmethod
    def _write_ranges(db, asset_id, ranges):
        from app.services.library.validation import validate_source_range
        for item in ranges:
            validate_source_range(item)
            db.execute("""INSERT INTO shot_records VALUES(?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET data=excluded.data""",
                       (item["id"], asset_id, item["start"], item["end"], json.dumps(item, ensure_ascii=False)))

    @staticmethod
    def _write_contents(db, asset_id, segments):
        revisions = {r["id"]: json.loads(r["data"]).get("revision") for r in
                     db.execute("SELECT id,data FROM shot_records WHERE asset_id=?", (asset_id,))}
        for item in segments:
            item = {**item, "range_revisions": {rid: revisions[rid] for rid in item["range_ids"]}}
            db.execute("INSERT INTO content_segments VALUES(?,?,?,?,?)",
                       (item["id"], asset_id, item["start"], item["end"], json.dumps(item, ensure_ascii=False)))
            db.executemany("INSERT INTO content_links VALUES(?,?)", [(item["id"], rid) for rid in item["range_ids"]])

    def replace_contents(self, library_id, asset_id, segments, expected_ranges):
        """One transaction publishes a consistent event-to-range index."""
        with self.database(library_id) as db:
            expected = set(expected_ranges)
            for row in db.execute("SELECT id FROM shot_records WHERE asset_id=?", (asset_id,)).fetchall():
                if row[0] not in expected:
                    db.execute("DELETE FROM shot_records WHERE id=?", (row[0],))
            db.execute("DELETE FROM content_segments WHERE asset_id=?", (asset_id,))
            self._write_contents(db, asset_id, segments)

    def analysis_snapshot(self, library_id, asset_ids, *, db=None):
        """Deterministic snapshot used by partial import and its transaction guard."""
        if db is None:
            with self.database(library_id) as connection:
                return self.analysis_snapshot(library_id, asset_ids, db=connection)
        selected = set(asset_ids)
        assets = {r["id"]: r for r in self._asset_rows(db) if r["id"] in selected}
        rows = {}
        for name, table in (("ranges", "shot_records"), ("contents", "content_segments")):
            rows[name] = {r["id"]: {**json.loads(r["data"]), "asset_id": r["asset_id"]} for r in db.execute(f"SELECT id,asset_id,data FROM {table} ORDER BY id")
                          if r["asset_id"] in selected}
        for item in rows["contents"].values():
            for field in ("library_id", "path", "episode", "fingerprint", "fps", "duration", "missing_range_ids", "stale_reasons"):
                item.pop(field, None)
        caches = {r[0]: json.loads(r[1]) for r in db.execute("""SELECT key,value FROM analysis_cache
            WHERE key='work' OR key LIKE 'analysis-generation:%' OR key LIKE 'agent-import:%' ORDER BY key""")}
        return {"assets": assets, **rows, "cache": caches}

    def apply_analysis_patch(self, library_id, asset_ids, expected_hash, patches, receipt_key, receipt):
        """Commit only reviewed ranges/events, retaining all unrelated rows and derivatives."""
        from app.services.library.runner import library_operation
        with library_operation(library_id), self.database(library_id) as db:
            db.execute("BEGIN IMMEDIATE")
            snapshot = self.analysis_snapshot(library_id, asset_ids, db=db)
            if stable_hash(snapshot) != expected_hash:
                raise ValueError("检查后素材库已有新修改，请重新检查结果；本批未写入。")
            for asset_id, patch in patches.items():
                if asset_id not in snapshot["assets"] or patch["status"] not in {"ready", "partial"}:
                    raise ValueError("导入范围或状态无效。")
                self._write_ranges(db, asset_id, patch["ranges"])
                for identifier in patch["remove_contents"]:
                    db.execute("DELETE FROM content_segments WHERE id=? AND asset_id=?", (identifier, asset_id))
                self._write_contents(db, asset_id, patch["contents"])
                db.execute("UPDATE assets SET status=? WHERE id=?", (patch["status"], asset_id))
            db.execute("INSERT OR REPLACE INTO analysis_cache VALUES(?,?)", (receipt_key, json.dumps(receipt, ensure_ascii=False)))

    def records(self, identifiers, kind="ranges", *, include_stale=False):
        """Read current evidence; the browser may also request retained, outdated records."""
        table = {"ranges": "shot_records", "contents": "content_segments"}[kind]
        result = []
        indices = {}
        assets = {(a["library_id"],a["id"]):a for a in self.assets(identifiers)}
        ranges = {(r["library_id"], r["id"]): r for r in self.records(identifiers, include_stale=True)} if kind == "contents" else {}
        for identifier in library_ids(identifiers):
            if self.schema(identifier) != SCHEMA_VERSION:
                continue
            with self.database(identifier) as db:
                rows = db.execute(f"""SELECT r.data,a.id AS asset_id,a.path,a.episode,a.fingerprint,a.fps,a.duration
                    FROM {table} r JOIN assets a ON a.id=r.asset_id ORDER BY a.episode,r.start""").fetchall()
                generations = {r[0].split(":",1)[1]:json.loads(r[1]) for r in db.execute(
                    "SELECT key,value FROM analysis_cache WHERE key LIKE 'analysis-generation:%'")}
            for row in rows:
                data = json.loads(row["data"])
                asset = assets[(identifier,row["asset_id"])]
                reasons = []
                if kind == "ranges":
                    if data.get("index_protocol") != FRAME_PROTOCOL:
                        reasons.append("旧解析需要按2.9.1重新建立源帧索引与拉片")
                    else:
                        from app.services.library.frame_extractor import source_index
                        from app.services.library.validation import validate_source_range
                        key = (identifier, row["asset_id"])
                        if key not in indices:
                            try:
                                indices[key] = source_index(asset, self.directory(identifier))
                            except (ValueError, OSError):
                                indices[key] = None
                        index = indices[key]
                        if index is None or any(data.get(k) != v for k, v in index.binding().items()):
                            reasons.append("源帧索引缺失或已更新，请重新拉片")
                        else:
                            try:
                                validate_source_range(data)
                            except ValueError as exc:
                                reasons.append(str(exc))
                    if data.get("fingerprint") != row["fingerprint"]:
                        reasons.append("原视频身份已改变")
                    if data.get("subtitle_identity") != asset.get("subtitle_identity"):
                        reasons.append("字幕登记已改变")
                    if data.get("generation", "") != generations.get(data["id"], ""):
                        data.update(analysis_state="uncertain", generation=generations.get(data["id"], ""))
                        reasons.append("已要求重新分析，尚未保存新观察")
                else:
                    missing = []
                    if set(data.get("range_revisions", {})) != set(data["range_ids"]):
                        reasons.append("内容段尚无完整观察修订记录，请继续本集内容分段")
                    for rid in data["range_ids"]:
                        item = ranges.get((identifier, rid))
                        if not item or item.get("stale_reasons") or (
                                rid in data.get("range_revisions", {}) and
                                data["range_revisions"][rid] != item.get("revision")):
                            missing.append(rid)
                    if missing:
                        reasons.append("关联观察缺失或已更新，请继续内容分段")
                    data["missing_range_ids"] = missing
                if reasons and not include_stale:
                    continue
                if include_stale:
                    data["stale_reasons"] = reasons
                    if kind == "ranges":
                        data["observed_fingerprint"] = data.get("fingerprint")
                result.append({**data, **{k: row[k] for k in row.keys() if k != "data"},
                               "library_id": identifier})
        return result

    def content_status(self, identifiers):
        """Derive UI availability from the same current evidence used by retrieval."""
        assets = self.assets(identifiers)
        ranges = self.records(identifiers, include_stale=True)
        contents = self.records(identifiers, "contents", include_stale=True)
        result = {}
        for asset in assets:
            own = [r for r in ranges if (r["library_id"], r["asset_id"]) == (asset["library_id"], asset["id"])]
            events = [e for e in contents if (e["library_id"], e["asset_id"]) == (asset["library_id"], asset["id"])]
            current = [r for r in own if not r["stale_reasons"]]
            valid = [e for e in events if not e["stale_reasons"]]
            linked = {rid for e in valid for rid in e["range_ids"]}
            usable_ids = {r["id"] for r in current if r["id"] in linked and r.get("analysis_state") == "ready" and
                          r.get("content_kind") == "story" and r["end"]-r["start"] >= 1.24}
            usable = len(usable_ids)
            stale = sum(bool(r["stale_reasons"]) for r in own)
            uncertain = sum(r.get("analysis_state") != "ready" for r in current)
            if stale or any(e["stale_reasons"] for e in events):
                label = "资料待更新"
            elif not current:
                label = "待分析"
            elif not valid or any(r["id"] not in linked for r in current):
                label = "观察已保存，目录待完成"
            elif uncertain:
                label = "目录已保存，观察待确认"
            elif asset["status"] != "ready":
                label = "部分结果已保存，继续拉片可补缺项"
            else:
                label = "拉片完成" if usable else "拉片完成，无可用剧情镜头"
            result[(asset["library_id"], asset["id"])] = {
                "label": label, "observations": len(current), "contents": len(valid),
                "usable": usable, "stale": stale, "uncertain": uncertain,
                "searchable_contents": sum(bool(usable_ids.intersection(e["range_ids"])) for e in valid),
                "incomplete": asset["status"] != "ready" or not current or bool(stale or uncertain or
                    any(e["stale_reasons"] for e in events) or any(r["id"] not in linked for r in current)),
            }
        return result

    def correct_range(self, library_id, range_id, changes, reason, *, expected_revision=None):
        """Corrections are local evidence revisions; unrelated observations stay valid."""
        from app.services.library.runner import library_operation
        if not reason.strip():
            raise ValueError("请说明更正依据。")
        with library_operation(library_id), self.database(library_id) as db:
            row = db.execute("SELECT data FROM shot_records WHERE id=?", (range_id,)).fetchone()
            if row is None:
                raise ValueError("镜头范围不存在。")
            item = json.loads(row[0])
            current = next((r for r in self.records(library_id) if r["id"] == range_id), None)
            if current is None or (expected_revision is not None and item.get("revision") != expected_revision):
                raise ValueError("该镜头或来源已更新，请刷新后再更正。")
            old = dict(item)
            from app.services.library.validation import correction_result
            item = correction_result(item, changes)
            item["correction"] = {"reason":reason.strip(),"time":time.time(),"fields":sorted(changes)}
            # Explicit human correction becomes the current observation, with its own revision.
            db.execute("INSERT OR REPLACE INTO analysis_cache VALUES(?,?)",
                       ("analysis-generation:"+range_id,json.dumps(item.get("generation",""))))
            item["revision"] = stable_hash({k:v for k,v in item.items() if k != "revision"})
            correction = {"before": old, "after": item, "reason": reason.strip(), "time": time.time()}
            db.execute("INSERT INTO analysis_cache VALUES(?,?)",
                       ("correction:"+uuid4().hex, json.dumps(correction, ensure_ascii=False)))
            db.execute("UPDATE shot_records SET data=? WHERE id=?", (json.dumps(item, ensure_ascii=False), range_id))
        return item

    def assets(self, library_id):
        result = []
        for identifier in library_ids(library_id):
            with self.database(identifier) as db:
                result.extend({**row, "library_id": identifier} for row in self._asset_rows(db))
        return result

    @staticmethod
    def _asset_rows(db):
        rows = db.execute("""SELECT a.*, s.source_path AS subtitle_path, s.original_path, s.processed_path, s.content_hash,
            s.settings AS subtitle_settings FROM assets a
            LEFT JOIN subtitle_sources s ON s.id=a.active_subtitle ORDER BY a.episode,a.path""").fetchall()
        result = []
        for row in rows:
            item = dict(row)
            settings = json.loads(item.get("subtitle_settings") or "{}")
            item["subtitle_identity"] = item["active_subtitle"] if settings.get("_processing") else item.get("content_hash")
            result.append(item)
        return result

    def upsert_asset(self, library_id, media, episode):
        with self.database(library_id) as db:
            row = db.execute("SELECT * FROM assets WHERE path=?", (media.path,)).fetchone()
            identifier = row["id"] if row else str(uuid4())
            status = row["status"] if row and row["fingerprint"] == media.fingerprint else "pending"
            db.execute("""INSERT INTO assets(id,path,episode,size,mtime_ns,fingerprint,duration,width,height,fps,status)
                VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET
                episode=excluded.episode,size=excluded.size,mtime_ns=excluded.mtime_ns,fingerprint=excluded.fingerprint,
                duration=excluded.duration,width=excluded.width,height=excluded.height,fps=excluded.fps,status=excluded.status""",
                (identifier, media.path, episode, media.size, media.mtime_ns, media.fingerprint,
                 media.duration, media.width, media.height, media.fps, status))
        return identifier

    def set_subtitle(self, library_id, asset_id, source_path, original_path, processed_path, content_hash, settings):
        identifier = stable_hash((asset_id, content_hash, settings))
        with self.database(library_id) as db:
            previous = db.execute("SELECT active_subtitle FROM assets WHERE id=?", (asset_id,)).fetchone()[0]
            db.execute("""INSERT INTO subtitle_sources VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET source_path=excluded.source_path""",
                       (identifier, asset_id, source_path, str(original_path), str(processed_path), content_hash, json.dumps(settings, ensure_ascii=False)))
            db.execute("UPDATE assets SET active_subtitle=?,status=CASE WHEN active_subtitle=? THEN status ELSE 'pending' END WHERE id=?",
                       (identifier, identifier, asset_id))
        return previous != identifier

    def restore_subtitle(self, library_id, asset, subtitle, manifest):
        """Update the registration only; retained observations become stale by identity."""
        self.require_content_library(library_id)
        with self.database(library_id) as db:
            db.execute("BEGIN IMMEDIATE")
            active = db.execute("SELECT active_subtitle,fingerprint FROM assets WHERE id=?", (asset["id"],)).fetchone()
            if not active or tuple(active) != (asset["active_subtitle"], asset["fingerprint"]):
                raise ValueError("素材或字幕登记已变化，请重新校验后恢复。")
            db.execute("INSERT OR IGNORE INTO subtitle_sources VALUES(?,?,?,?,?,?,?)", (
                subtitle["id"], asset["id"], subtitle["source_path"], subtitle["original_path"],
                subtitle["processed_path"], subtitle["content_hash"], json.dumps(subtitle["settings"], ensure_ascii=False)))
            db.execute("UPDATE assets SET active_subtitle=?,status='pending' WHERE id=?", (subtitle["id"], asset["id"]))
            committed = {**manifest, "state": "committed", "schema": SCHEMA_VERSION,
                         "old_status": asset["status"], "new_subtitle": subtitle}
            db.execute("INSERT OR REPLACE INTO analysis_cache VALUES(?,?)",
                       ("subtitle-migration:" + manifest["id"], json.dumps(committed, ensure_ascii=False)))
        return committed

    def rebind(self, library_id, asset_id, media):
        from app.services.library.runner import library_operation
        with library_operation(library_id):
            return self._rebind(library_id, asset_id, media)

    def _rebind(self, library_id, asset_id, media):
        asset = next(item for item in self.assets(library_id) if item["id"] == asset_id)
        if asset["fingerprint"] != media.fingerprint or asset["size"] != media.size or abs(asset["duration"] - media.duration) > .1:
            raise ValueError("新路径与原素材指纹或时长不一致，请作为新素材导入。")
        with self.database(library_id) as db:
            existing = db.execute("SELECT id FROM assets WHERE path=? AND id<>?", (media.path, asset_id)).fetchone()
            if existing:
                raise ValueError("这个路径已登记为另一条素材，请先确认素材对应关系。")
            db.execute("UPDATE assets SET path=?,mtime_ns=? WHERE id=?", (media.path, media.mtime_ns, asset_id))

    def set_asset_status(self, library_id, asset_id, status):
        with self.database(library_id) as db:
            db.execute("UPDATE assets SET status=? WHERE id=?", (status, asset_id))

    def cache_get(self, library_id, key):
        with self.database(library_id) as db:
            row = db.execute("SELECT value FROM analysis_cache WHERE key=?", (key,)).fetchone()
        try:
            return json.loads(row[0]) if row else None
        except (ValueError, TypeError):
            return None

    def cache_put(self, library_id, key, value):
        with self.database(library_id) as db:
            db.execute("INSERT OR REPLACE INTO analysis_cache VALUES(?,?)", (key, json.dumps(value, ensure_ascii=False)))

    def cache_delete(self, library_id, key):
        with self.database(library_id) as db:
            db.execute("DELETE FROM analysis_cache WHERE key=?", (key,))

    def job(self, library_id, identifier="build", **updates):
        with self.database(library_id) as db:
            if updates:
                db.execute("""INSERT INTO analysis_jobs(id,state,progress,message,updated,settings) VALUES(?,?,?,?,?,?)
                    ON CONFLICT(id) DO UPDATE SET state=excluded.state,progress=excluded.progress,message=excluded.message,updated=excluded.updated,settings=excluded.settings""",
                    (identifier, updates.get("state", "running"), updates.get("progress", 0), updates.get("message", ""), time.time(), json.dumps(updates.get("settings", {}), ensure_ascii=False)))
            row = db.execute("SELECT * FROM analysis_jobs WHERE id=?", (identifier,)).fetchone()
        return dict(row) if row else None

    def record_analysis_diagnostic(self, library_id, context, diagnostic):
        identity = (context.get("asset_id"), context.get("stage"), context.get("window"))
        if context.get("group"):
            identity += (context["group"],)
        identifier = "error:" + stable_hash(identity)[:32]
        if diagnostic["state"] == "resolved":
            with self.database(library_id) as db:
                db.execute("UPDATE analysis_jobs SET state='resolved',updated=? WHERE id=?", (time.time(), identifier))
            return
        record = {**context, **diagnostic, "time": time.time()}
        message = (f"第 {context.get('episode', '?')} 集，{context.get('stage', '')}，"
                   f"窗口 {context.get('window', '?')}（{context.get('start', 0):.2f}–{context.get('end', 0):.2f}秒），"
                   f"第 {diagnostic.get('attempt', 1)} 次尝试：{diagnostic.get('error', '')}")
        self.job(library_id, identifier, state="failed", message=message, settings=record)
        with (self.directory(library_id) / "errors.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(record, ensure_ascii=False) + "\n")

    def analysis_errors(self, library_id):
        with self.database(library_id) as db:
            rows = db.execute("SELECT * FROM analysis_jobs WHERE id LIKE 'error:%' AND state='failed' ORDER BY updated DESC").fetchall()
        return [{**dict(row), "details": json.loads(row["settings"])} for row in rows]

    def resolve_analysis_errors(self, library_id, asset_id, stage=None, window=None):
        with self.database(library_id) as db:
            rows = db.execute("SELECT id,settings FROM analysis_jobs WHERE id LIKE 'error:%' AND state='failed'").fetchall()
            for row in rows:
                details = json.loads(row["settings"])
                if details.get("asset_id") == asset_id and (stage is None or details.get("stage") == stage) and (window is None or details.get("window") == window):
                    db.execute("UPDATE analysis_jobs SET state='resolved',updated=? WHERE id=?", (time.time(), row["id"]))

    def deletion_info(self, library_id):
        directory = self.directory(library_id)
        with self.database(library_id) as db:
            counts = {name: db.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                      for name in ("assets", "scenes", "analysis_cache")}
            running_records = db.execute("SELECT COUNT(*) FROM analysis_jobs WHERE id IN ('build','import','subtitle','files','agent') AND state='running'").fetchone()[0]
            if db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION:
                counts["scenes"] = db.execute("SELECT COUNT(*) FROM content_segments").fetchone()[0]
        return {"directory": str(directory), **counts, "running_records": running_records}

    def delete_library(self, library_id, confirmation_name, interrupted_confirmed=False):
        from app.services.library.runner import library_operation
        with library_operation(library_id):
            return self._delete_library(library_id, confirmation_name, interrupted_confirmed)

    def _delete_library(self, library_id, confirmation_name, interrupted_confirmed=False):
        """Remove this library's derived data after validating exact path boundaries."""
        from app.services.library import runner
        from app.services.narration_task import list_tasks, task_directory

        if self.root.is_symlink() or self.root.is_junction():
            raise ValueError("素材库根目录是链接，拒绝执行删除。")
        identifier = str(UUID(str(library_id)))
        library = next((item for item in self.list_libraries() if item["id"] == identifier), None)
        if not library or str(confirmation_name).strip() != library["name"]:
            raise ValueError("请输入要删除的素材库名称确认。")
        tasks = [task for task in list_tasks() if identifier in library_ids(task)]
        if runner.running("library:" + identifier) or any(runner.running("narration:" + task["id"]) for task in tasks):
            raise ValueError("素材库或关联的生成任务仍在运行，请先停止并等待任务结束。")
        info = self.deletion_info(identifier)
        if (info["running_records"] or any(task.get("state") == "running" for task in tasks)) and not interrupted_confirmed:
            raise ValueError("存在尚未结束的任务记录，请先确认旧程序已经停止。")

        root = self.root.resolve(strict=True)
        logical = self.root / identifier
        if logical.is_symlink() or logical.is_junction():
            raise ValueError("素材库目录是链接，拒绝删除。")
        target = logical.resolve(strict=True)
        if target.parent != root or target.name != identifier or not target.is_dir():
            raise ValueError("素材库目录不在预期的独立目录内，拒绝删除。")
        for folder, directories, filenames in os.walk(target, followlinks=False):
            for name in directories + filenames:
                entry = Path(folder) / name
                if entry.is_symlink() or entry.is_junction():
                    raise ValueError("素材库包含链接，拒绝递归删除。")
        tombstone = root / (".deleting-" + identifier + "-" + uuid4().hex)
        if tombstone.resolve().parent != root:
            raise ValueError("临时删除目录不在素材库根目录内。")
        target.rename(tombstone)
        try:
            with self.connection(root / "catalog.sqlite") as db:
                db.execute("DELETE FROM libraries WHERE id=?", (identifier,))
        except Exception:
            tombstone.rename(target)
            raise

        cleanup_message = ""
        try:
            # Both absolute paths were checked before moving/removing a directory.
            if tombstone.resolve(strict=True).parent != root:
                raise ValueError("删除目录越界。")
            shutil.rmtree(tombstone)
        except OSError:
            cleanup_message = f"素材库已从列表移除，部分缓存未清理，请关闭占用后清理：{tombstone}"
        for task in tasks:
            if task.get("state") != "complete":
                try:
                    state_path = task_directory(task["id"]) / "state.json"
                    state = read_json(state_path)
                    state.update(state="failed", message="关联素材库已删除，未完成任务无法继续。")
                    write_json(state_path, state)
                except (OSError, ValueError):
                    cleanup_message += "\n部分旧任务状态未更新；其文件仍保留，但已删除素材库的未完成任务无法继续。"
        return cleanup_message

    def remove_assets(self, library_id, asset_ids, confirmed=False, interrupted_confirmed=False):
        from app.services.library.runner import library_operation
        with library_operation(library_id):
            return self._remove_assets(library_id, asset_ids, confirmed, interrupted_confirmed)

    def _remove_assets(self, library_id, asset_ids, confirmed=False, interrupted_confirmed=False):
        """Remove selected registrations and owned derivatives, never original media."""
        from app.services.library import runner
        from app.services.narration_task import list_tasks

        if not confirmed or not asset_ids:
            raise ValueError("请勾选并确认要移除的素材。")
        if self.root.is_symlink() or self.root.is_junction():
            raise ValueError("素材库根目录是链接，拒绝删除派生目录。")
        identifiers = list(dict.fromkeys(str(UUID(str(value))) for value in asset_ids))
        registered = {asset["id"]:asset for asset in self.assets(library_id)}
        known = set(registered)
        if not set(identifiers).issubset(known):
            raise ValueError("所选素材不属于此素材库。")
        tasks = [task for task in list_tasks() if library_id in library_ids(task)]
        if runner.running("library:" + library_id) or any(runner.running("narration:" + task["id"]) for task in tasks):
            raise ValueError("请先停止建库和关联的生成任务。")
        info = self.deletion_info(library_id)
        if (info["running_records"] or any(task.get("state") == "running" for task in tasks)) and not interrupted_confirmed:
            raise ValueError("请确认旧程序及其他窗口没有运行关联任务。")
        library_root = self.directory(library_id).resolve(strict=True)
        moved = []
        try:
            for identifier in identifiers:
                for category in ("frames", "subtitles"):
                    parent = library_root / category
                    if parent.is_symlink() or parent.is_junction():
                        raise ValueError("派生目录是链接，拒绝移除。")
                    owned_name = registered[identifier]["fingerprint"] if category == "frames" and self.schema(library_id) == SCHEMA_VERSION else identifier
                    logical = parent / owned_name
                    if not logical.exists():
                        continue
                    if logical.is_symlink() or logical.is_junction():
                        raise ValueError("素材派生目录是链接，拒绝移除。")
                    target = logical.resolve(strict=True)
                    if target.parent != parent.resolve() or target.name != owned_name or not target.is_dir():
                        raise ValueError("素材派生目录越界，拒绝移除。")
                    for folder, directories, files in os.walk(target, followlinks=False):
                        if any((Path(folder) / name).is_symlink() or (Path(folder) / name).is_junction() for name in directories + files):
                            raise ValueError("素材派生目录包含链接，拒绝移除。")
                    tombstone = library_root / (".removing-" + uuid4().hex)
                    if tombstone.resolve().parent != library_root:
                        raise ValueError("临时移除目录越界。")
                    target.rename(tombstone)
                    moved.append((target, tombstone))
            with self.database(library_id) as db:
                for identifier in identifiers:
                    if db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION:
                        db.execute("DELETE FROM content_segments WHERE asset_id=?",(identifier,))
                        db.execute("DELETE FROM shot_records WHERE asset_id=?",(identifier,))
                    db.execute("DELETE FROM shots WHERE scene_id IN (SELECT id FROM scenes WHERE asset_id=?)", (identifier,))
                    db.execute("DELETE FROM scenes WHERE asset_id=?", (identifier,))
                    db.execute("DELETE FROM subtitle_sources WHERE asset_id=?", (identifier,))
                    db.execute("DELETE FROM assets WHERE id=?", (identifier,))
                    db.execute("DELETE FROM analysis_cache WHERE key=?", ("episode-summary:" + identifier,))
                records = db.execute("SELECT id,settings FROM analysis_jobs WHERE id LIKE 'error:%'").fetchall()
                for record in records:
                    if json.loads(record["settings"]).get("asset_id") in identifiers:
                        db.execute("DELETE FROM analysis_jobs WHERE id=?", (record["id"],))
        except Exception:
            for original, temporary in reversed(moved):
                temporary.rename(original)
            raise
        remaining = []
        for original, temporary in moved:
            if temporary.resolve(strict=True).parent != library_root:
                raise ValueError("清理目录越界。")
            try:
                shutil.rmtree(temporary)
            except OSError:
                remaining.append(str(temporary))
        # Shared model-result cache entries have no reliable per-asset ownership.
        # Keep them for reimport; retrieval only uses scenes linked to registered assets.
        return remaining
