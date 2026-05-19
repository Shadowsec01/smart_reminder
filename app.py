# =============================================================================
#  app.py  —  SmartTask Pro | Main Flask Application
#  Author  : [Your Full Name] | Reg No: [Your Reg Number] | Dept: [Department]
#
#  FRAMEWORK: Flask (micro web framework for Python)
#  DATABASE : SQLite via Flask-SQLAlchemy ORM
#  SCHEDULER: APScheduler (background job scheduling for reminders)
#  AI MODULE: ai_engine.py (all AI logic is isolated there)
#
#  ARCHITECTURE PATTERN: MVC (Model-View-Controller)
#    • Model      → Task and Reminder SQLAlchemy classes (database schema)
#    • View       → templates/index.html (rendered by Jinja2)
#    • Controller → Flask route functions (@app.route decorators)
#
#  REST API DESIGN:
#    The backend exposes a JSON REST API that the frontend consumes via
#    JavaScript fetch() calls.  This makes the system easy to extend
#    (mobile app, external integrations) without changing the backend.
#
#  HOW TO RUN:
#    pip install -r requirements.txt
#    python app.py
#    → Open http://127.0.0.1:5000
# =============================================================================

# ─── Standard library imports ─────────────────────────────────────────────────
from datetime import datetime          # Date/time manipulation
import os                              # File path helpers
import logging                         # Production-grade logging

# ─── Third-party imports ──────────────────────────────────────────────────────
from flask import Flask, render_template, request, jsonify, abort
from flask_sqlalchemy import SQLAlchemy          # ORM (Object-Relational Mapper)
from apscheduler.schedulers.background import BackgroundScheduler
# APScheduler runs background jobs (reminder checks) without blocking Flask

# ─── Local AI module ──────────────────────────────────────────────────────────
from ai_engine import AIEngine

# =============================================================================
# APPLICATION SETUP
# =============================================================================

# Flask app factory
# __name__ tells Flask where to find templates/ and static/ folders
app = Flask(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────
# SQLite database stored in the same directory as app.py
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"sqlite:///{os.path.join(BASE_DIR, 'smarttask.db')}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False   # Suppress warning
app.config["SECRET_KEY"] = "smarttask-secret-2025"     # CSRF / session signing

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Database and AI initialisation ──────────────────────────────────────────
db = SQLAlchemy(app)      # SQLAlchemy ORM bound to this Flask app
ai = AIEngine()           # Instantiate AI engine (one shared instance)

# ─── Background Scheduler ─────────────────────────────────────────────────────
# BackgroundScheduler runs in a daemon thread alongside Flask.
# It fires reminder jobs at scheduled times without blocking the web server.
scheduler = BackgroundScheduler()
scheduler.start()
logger.info("Background scheduler started.")


# =============================================================================
# DATABASE MODELS  (M in MVC)
# =============================================================================

class Task(db.Model):
    """
    ORM Model: represents one row in the 'task' SQLite table.

    SQLAlchemy maps Python class attributes → database columns automatically.
    CRUD operations (Create, Read, Update, Delete) are done via ORM methods,
    making the code database-agnostic (SQLite now, PostgreSQL later — same code).
    """

    __tablename__ = "task"

    # Primary key — auto-incremented integer unique identifier
    id = db.Column(db.Integer, primary_key=True)

    # Core task fields
    title       = db.Column(db.String(300), nullable=False)
    description = db.Column(db.Text, default="")

    # AI-predicted metadata (filled by ai_engine.py at creation time)
    category       = db.Column(db.String(60),  default="General")  # Rule-Based AI
    priority       = db.Column(db.String(20),  default="Medium")   # Heuristic AI
    priority_score = db.Column(db.Float,       default=5.0)        # Numeric score

    # Time fields
    due_date      = db.Column(db.DateTime, nullable=True)   # Extracted by NLP
    reminder_time = db.Column(db.DateTime, nullable=True)   # Suggested by AI

    # Status flags
    is_completed = db.Column(db.Boolean, default=False)
    is_reminded  = db.Column(db.Boolean, default=False)   # Has reminder fired?

    # Audit timestamps
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)

    # Comma-separated tags string (e.g. "#urgent,#school")
    tags = db.Column(db.String(400), default="")

    # Student who owns this task (stored for display purposes)
    student_name = db.Column(db.String(150), default="")
    reg_number   = db.Column(db.String(50),  default="")
    department   = db.Column(db.String(100), default="")

    def to_dict(self) -> dict:
        """
        Serialise the ORM object to a plain Python dict.
        This dict is then converted to JSON by Flask's jsonify().

        Why a method? Because SQLAlchemy model objects are not directly
        JSON-serialisable — we must convert datetimes, split tag strings, etc.
        """
        return {
            "id":             self.id,
            "title":          self.title,
            "description":    self.description,
            "category":       self.category,
            "priority":       self.priority,
            "priority_score": self.priority_score,
            "due_date":       self.due_date.isoformat() if self.due_date else None,
            "reminder_time":  self.reminder_time.isoformat() if self.reminder_time else None,
            "is_completed":   self.is_completed,
            "is_reminded":    self.is_reminded,
            "created_at":     self.created_at.isoformat(),
            "completed_at":   self.completed_at.isoformat() if self.completed_at else None,
            "tags":           [t for t in self.tags.split(",") if t],
            "student_name":   self.student_name,
            "reg_number":     self.reg_number,
            "department":     self.department,
        }


# Create all tables in the database (idempotent — safe to call multiple times)
with app.app_context():
    db.create_all()
    logger.info("Database tables created/verified.")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def schedule_reminder(task: Task) -> None:
    """
    Register a one-off APScheduler job to fire when the reminder time arrives.

    APScheduler 'date' trigger: fires exactly once at a specific datetime.
    The job calls send_reminder(task_id) which marks the task as reminded.
    If a reminder already exists for this task, it is safely replaced.
    """
    if not task.reminder_time or task.reminder_time <= datetime.utcnow():
        return   # Already past — no point scheduling

    job_id = f"reminder_{task.id}"

    # Remove existing job if present (e.g. task was edited)
    if scheduler.get_job(job_id):
        scheduler.remove_job(job_id)

    scheduler.add_job(
        func=send_reminder,
        trigger="date",
        run_date=task.reminder_time,
        args=[task.id],
        id=job_id,
        replace_existing=True,
    )
    logger.info(f"Reminder scheduled for task {task.id} at {task.reminder_time}")


def send_reminder(task_id: int) -> None:
    """
    Background job payload: marks a task as 'reminded'.
    Runs in the scheduler thread, so it needs its own app context.
    The frontend polls /api/reminders/active to pick up triggered reminders.
    """
    with app.app_context():
        task = Task.query.get(task_id)
        if task and not task.is_completed:
            task.is_reminded = True
            db.session.commit()
            logger.info(f"Reminder fired for task {task.id}: '{task.title}'")


# =============================================================================
# ROUTES  (C in MVC — Controllers)
# =============================================================================

# ─── Main page ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """
    Serve the single-page application.
    Jinja2 renders templates/index.html and returns it as HTML.
    All subsequent communication is via JSON API calls from JavaScript.
    """
    return render_template("index.html")


# ─── TASK ENDPOINTS ───────────────────────────────────────────────────────────

@app.route("/api/tasks", methods=["GET"])
def get_tasks():
    """
    GET /api/tasks
    Returns all tasks, optionally filtered by status and/or category.

    Query params:
      ?status=all|active|completed
      ?category=all|Study|Work|Health|...
      ?sort=priority|created|due_date

    Ordering: priority_score DESC (highest priority first),
              then created_at DESC (newest first within same priority).
    """
    status   = request.args.get("status",   "all")
    category = request.args.get("category", "all")
    sort_by  = request.args.get("sort",     "priority")

    query = Task.query

    # Apply status filter
    if status == "active":
        query = query.filter_by(is_completed=False)
    elif status == "completed":
        query = query.filter_by(is_completed=True)

    # Apply category filter
    if category != "all":
        query = query.filter_by(category=category)

    # Apply sort order
    if sort_by == "due_date":
        query = query.order_by(Task.due_date.asc().nullslast())
    elif sort_by == "created":
        query = query.order_by(Task.created_at.desc())
    else:
        # Default: sort by AI priority score, highest first
        query = query.order_by(Task.priority_score.desc(), Task.created_at.desc())

    tasks = query.all()
    return jsonify([t.to_dict() for t in tasks])


@app.route("/api/tasks", methods=["POST"])
def create_task():
    """
    POST /api/tasks
    Create a new task with full AI processing pipeline:

    Pipeline steps:
      1. NLP parse  → extract due_date, clean title
      2. Rule-Based → predict category
      3. Heuristic  → predict priority + score
      4. Rule-Based → suggest reminder time
      5. Pattern    → extract tags
      6. Save to DB
      7. Schedule reminder job

    Returns:
      201 Created with task dict and ai_analysis metadata so the frontend
      can display what the AI detected (transparent AI).
    """
    data = request.get_json(force=True)
    if not data or not data.get("title"):
        abort(400, description="'title' field is required.")

    raw_input   = data["title"].strip()
    description = data.get("description", "")

    # ── AI PIPELINE ──────────────────────────────────────────────────────────
    # Step 1: NLP — parse dates and clean the title
    parsed      = ai.parse_natural_language(raw_input)
    due_date    = parsed["due_date"]
    clean_title = parsed["clean_title"]

    # Step 2: Rule-Based AI — classify the task into a category
    category = ai.predict_category(raw_input)

    # Step 3: Heuristic AI — score and label priority
    priority, priority_score = ai.predict_priority(raw_input, due_date)

    # Step 4: Rule-Based Decision — suggest when to send reminder
    reminder_time = ai.suggest_reminder(due_date, priority)

    # Step 5: Pattern Matching — extract tags
    tags = ai.extract_tags(raw_input)
    # ─────────────────────────────────────────────────────────────────────────

    # Build and persist the Task ORM object
    task = Task(
        title          = clean_title,
        description    = description,
        category       = category,
        priority       = priority,
        priority_score = priority_score,
        due_date       = due_date,
        reminder_time  = reminder_time,
        tags           = ",".join(tags),
        student_name   = data.get("student_name", ""),
        reg_number     = data.get("reg_number", ""),
        department     = data.get("department", ""),
    )

    db.session.add(task)
    db.session.commit()

    # Schedule the background reminder job
    schedule_reminder(task)

    logger.info(f"Task created: id={task.id}, priority={priority}, category={category}")

    return jsonify({
        "task": task.to_dict(),
        # Return AI analysis so the frontend can show what the AI detected
        "ai_analysis": {
            "detected_date":       due_date.isoformat() if due_date else None,
            "clean_title":         clean_title,
            "category":            category,
            "priority":            priority,
            "priority_score":      priority_score,
            "reminder_suggestion": reminder_time.isoformat() if reminder_time else None,
            "tags":                tags,
        },
    }), 201


@app.route("/api/tasks/<int:task_id>", methods=["GET"])
def get_task(task_id: int):
    """GET /api/tasks/<id> — Retrieve a single task by ID."""
    task = db.get_or_404(Task, task_id)
    return jsonify(task.to_dict())


@app.route("/api/tasks/<int:task_id>", methods=["PUT"])
def update_task(task_id: int):
    """
    PUT /api/tasks/<id>
    Update fields of an existing task.
    If the due_date or title is changed, re-run the relevant AI steps.
    """
    task = db.get_or_404(Task, task_id)
    data = request.get_json(force=True)

    if "title" in data:
        task.title = data["title"].strip()
        # Re-run AI categorisation and tag extraction on new title
        task.category = ai.predict_category(task.title)
        task.tags     = ",".join(ai.extract_tags(task.title))

    if "description" in data:
        task.description = data["description"]

    if "due_date" in data:
        if data["due_date"]:
            task.due_date = datetime.fromisoformat(data["due_date"])
        else:
            task.due_date = None
        # Re-compute priority and reminder with new due date
        task.priority, task.priority_score = ai.predict_priority(
            task.title, task.due_date
        )
        task.reminder_time = ai.suggest_reminder(task.due_date, task.priority)
        schedule_reminder(task)   # Re-schedule reminder

    if "priority" in data:
        # Allow manual priority override
        task.priority = data["priority"]

    if "category" in data:
        task.category = data["category"]

    if "is_completed" in data:
        task.is_completed = bool(data["is_completed"])
        task.completed_at = datetime.utcnow() if task.is_completed else None

    db.session.commit()
    logger.info(f"Task {task_id} updated.")
    return jsonify(task.to_dict())


@app.route("/api/tasks/<int:task_id>", methods=["DELETE"])
def delete_task(task_id: int):
    """
    DELETE /api/tasks/<id>
    Remove task from DB and cancel its scheduled reminder (if any).
    """
    task = db.get_or_404(Task, task_id)

    # Cancel the APScheduler reminder job before deleting the task
    job_id = f"reminder_{task_id}"
    if scheduler.get_job(job_id):
        scheduler.remove_job(job_id)

    db.session.delete(task)
    db.session.commit()
    logger.info(f"Task {task_id} deleted.")
    return jsonify({"message": "Task deleted successfully."})


@app.route("/api/tasks/<int:task_id>/toggle", methods=["POST"])
def toggle_task(task_id: int):
    """
    POST /api/tasks/<id>/toggle
    Toggle the completion status of a task.
    Completed tasks don't need their reminder to fire — cancel it.
    """
    task = db.get_or_404(Task, task_id)
    task.is_completed = not task.is_completed
    task.completed_at = datetime.utcnow() if task.is_completed else None

    if task.is_completed:
        # Cancel reminder for completed tasks
        job_id = f"reminder_{task_id}"
        if scheduler.get_job(job_id):
            scheduler.remove_job(job_id)

    db.session.commit()
    return jsonify(task.to_dict())


# ─── REMINDER ENDPOINTS ───────────────────────────────────────────────────────

@app.route("/api/reminders/active", methods=["GET"])
def get_active_reminders():
    """
    GET /api/reminders/active
    Return tasks that have been flagged as 'reminded' (scheduler fired)
    but have not yet been acknowledged by the user.

    The frontend polls this endpoint every 30 seconds to show popup notifications.
    """
    triggered = Task.query.filter(
        Task.is_reminded == True,       # noqa: E712
        Task.is_completed == False,     # noqa: E712
    ).all()

    return jsonify([t.to_dict() for t in triggered])


@app.route("/api/reminders/<int:task_id>/acknowledge", methods=["POST"])
def acknowledge_reminder(task_id: int):
    """
    POST /api/reminders/<id>/acknowledge
    Mark a reminder as acknowledged — prevents repeated popup notifications.
    """
    task = db.get_or_404(Task, task_id)
    task.is_reminded = False   # Reset flag after user sees it
    db.session.commit()
    return jsonify({"message": "Reminder acknowledged."})


@app.route("/api/reminders/upcoming", methods=["GET"])
def get_upcoming_reminders():
    """
    GET /api/reminders/upcoming
    Return active tasks due within the next 24 hours, sorted by urgency.
    Used to populate the 'Upcoming' sidebar panel.
    """
    from datetime import timedelta
    now = datetime.utcnow()
    cutoff = now + timedelta(hours=24)

    upcoming = Task.query.filter(
        Task.is_completed == False,       # noqa: E712
        Task.due_date.isnot(None),
        Task.due_date <= cutoff,
        ).order_by(Task.due_date.asc()).all()

    result = []
    for task in upcoming:
        delta = (task.due_date - now).total_seconds()
        urgency = (
            "overdue"  if delta < 0        else
            "critical" if delta < 3_600    else   # < 1 hour
            "high"     if delta < 10_800   else   # < 3 hours
            "medium"   if delta < 43_200   else   # < 12 hours
            "normal"
        )
        result.append({**task.to_dict(), "urgency": urgency, "seconds_remaining": delta})

    return jsonify(result)


# ─── STATISTICS ENDPOINT ──────────────────────────────────────────────────────

@app.route("/api/stats", methods=["GET"])
def get_stats():
    """
    GET /api/stats
    Aggregated task statistics for the dashboard.
    Uses SQLAlchemy aggregate functions (COUNT, GROUP BY) — efficient SQL.
    Also triggers NLG (Natural Language Generation) for an insight message.
    """
    total     = Task.query.count()
    completed = Task.query.filter_by(is_completed=True).count()
    active    = Task.query.filter_by(is_completed=False).count()
    overdue   = Task.query.filter(
        Task.is_completed == False,     # noqa: E712
        Task.due_date.isnot(None),
        Task.due_date < datetime.utcnow(),
        ).count()

    # Category breakdown using SQL GROUP BY
    cat_rows = (
        db.session.query(Task.category, db.func.count(Task.id))
        .group_by(Task.category)
        .all()
    )
    categories = {cat: cnt for cat, cnt in cat_rows}

    # Priority breakdown
    pri_rows = (
        db.session.query(Task.priority, db.func.count(Task.id))
        .group_by(Task.priority)
        .all()
    )
    priorities = {pri: cnt for pri, cnt in pri_rows}

    stats_dict = {
        "total":           total,
        "completed":       completed,
        "active":          active,
        "overdue":         overdue,
        "completion_rate": round((completed / total * 100) if total else 0, 1),
        "categories":      categories,
        "priorities":      priorities,
    }

    # AI NLG — generate a personalised insight message
    stats_dict["insight"] = ai.generate_insight(stats_dict)

    return jsonify(stats_dict)


# ─── AI ANALYSIS ENDPOINT (live preview) ─────────────────────────────────────

@app.route("/api/ai/analyze", methods=["POST"])
def analyze_text():
    """
    POST /api/ai/analyze
    Runs the AI pipeline on raw text WITHOUT saving to DB.
    Used by the frontend to show live AI predictions as the user types.
    This is the 'AI Preview' feature — instant feedback, no commitment.
    """
    data = request.get_json(force=True)
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400

    parsed        = ai.parse_natural_language(text)
    category      = ai.predict_category(text)
    priority, score = ai.predict_priority(text, parsed["due_date"])
    tags          = ai.extract_tags(text)
    reminder      = ai.suggest_reminder(parsed["due_date"], priority)

    return jsonify({
        "clean_title":        parsed["clean_title"],
        "detected_date":      parsed["due_date"].strftime("%a, %d %b %Y %H:%M") if parsed["due_date"] else None,
        "category":           category,
        "priority":           priority,
        "priority_score":     score,
        "tags":               tags,
        "reminder_suggestion": reminder.strftime("%a, %d %b %Y %H:%M") if reminder else None,
    })


# ─── STUDENT PROFILE ENDPOINT ─────────────────────────────────────────────────

@app.route("/api/student", methods=["GET", "POST"])
def student_profile():
    """
    GET  /api/student  → retrieve saved student profile from DB (last entry)
    POST /api/student  → save student profile; used to pre-fill task forms
    """
    if request.method == "GET":
        last = Task.query.filter(Task.student_name != "").first()
        if last:
            return jsonify({
                "student_name": last.student_name,
                "reg_number":   last.reg_number,
                "department":   last.department,
            })
        return jsonify({})

    data = request.get_json(force=True)
    # Profile is stored per-session in localStorage on the frontend;
    # this endpoint is just a passthrough validator.
    return jsonify({
        "student_name": data.get("student_name", ""),
        "reg_number":   data.get("reg_number", ""),
        "department":   data.get("department", ""),
        "saved":        True,
    })


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": str(e)}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Resource not found."}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error."}), 500


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # debug=True enables:
    #   • Auto-reload on code change
    #   • Detailed error pages in browser
    # Set debug=False in production!
    app.run(debug=True, host="0.0.0.0", port=5000)