"""Laya CLI — primary interface for humans, scripts, and systemd lifecycle.

Entry point: laya = "laya_server.cli:cli"

Subcommands:
  serve     Start the HTTP server (foreground)
  query     Hit /system_one (inference)
  health    Check service health
  stats     Show service statistics
  unload    Unload model from GPU
  version   Print version
  start     systemctl start laya-service        (system unit, root for start)
  stop      systemctl stop laya-service
  restart   systemctl restart laya-service
  status    Combined systemd + health state
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys

import click

from . import __version__
from .client import health, resolve_endpoint, stats, system_one, unload


def _parse_input(value: str):
    """Parse a JSON string or file path. If starts with { or [, try JSON; else read file."""
    stripped = value.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return json.loads(stripped)
    try:
        with open(stripped) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise click.ClickException(f"Cannot parse input '{value}': {e}")


def _output(data: dict, as_json: bool) -> None:
    """Output data as JSON or formatted text."""
    if as_json:
        click.echo(json.dumps(data, indent=2))
    else:
        _print_health(data)


def _print_health(data: dict) -> None:
    """Human-friendly health output."""
    status = data.get("status", "unknown")
    icon = {"ok": "●", "degraded": "○"}.get(status, "?")
    click.echo(f"{icon} Laya service: {status}")
    if data.get("model_loaded"):
        click.echo(f"  Model: loaded ({', '.join(data.get('providers', []))})")
    else:
        click.echo("  Model: NOT loaded")
    click.echo(f"  Uptime: {data.get('uptime_s', 0)}s")
    click.echo(f"  Version: {data.get('version', '?')}")


def _print_stats(data: dict) -> None:
    """Human-friendly stats output."""
    click.echo(f"Requests: {data.get('total_requests', 0)}")
    lat = data.get("latency_ms", {})
    if lat:
        click.echo(
            f"Latency: p50={lat.get('p50', 0)}ms "
            f"p99={lat.get('p99', 0)}ms last={lat.get('last', 0)}ms"
        )
    cache = data.get("cache", {})
    if cache:
        click.echo(
            f"Cache: {cache.get('hits', 0)} hits / "
            f"{cache.get('misses', 0)} misses (enabled: {cache.get('enabled')})"
        )
    click.echo(f"Model: {'loaded' if data.get('model_loaded') else 'not loaded'}")
    click.echo(f"Version: {data.get('version', '?')}")


def _print_answers(data: dict) -> None:
    """Human-friendly inference answer output."""
    answers = data.get("answers", {})
    for qid, answer in answers.items():
        qtype = answer.get("type", "?")
        click.echo(f"\n{qid} ({qtype}):")
        if qtype == "choice":
            click.echo(f"  → {answer.get('choice', '?')}")
            for label, prob in answer.get("probabilities", {}).items():
                bar = "█" * int(prob * 30)
                click.echo(f"    {label}: {prob:.4f} {bar}")
            click.echo(f"  confidence: {answer.get('confidence', 0):.4f}")
        elif qtype == "score":
            click.echo(f"  score: {answer.get('score', 0):.4f}")
            for lvl, prob in answer.get("probabilities", {}).items():
                bar = "█" * int(prob * 30)
                click.echo(f"    {lvl}: {prob:.4f} {bar}")
            click.echo(f"  confidence: {answer.get('confidence', 0):.4f}")
        elif qtype == "noul":
            click.echo(f"  noul: {answer.get('noul', 0):.4f}")
        routing = answer.get("rl_agent", {})
        if routing:
            click.echo(
                f"  routing: act={routing.get('act_probability', 0):.4f} "
                f"escalate={routing.get('escalate_probability', 0):.4f}"
            )
            if routing.get("experimental"):
                click.echo("  (experimental)")
    if data.get("cached"):
        click.echo("\n(cached)")


def _run_systemctl(args: list[str]) -> tuple[int, str]:
    """Run a systemctl command (system unit). Returns (returncode, output).

    The laya-service unit is installed as a SYSTEM unit (/etc/systemd/system/),
    so no --user flag is used. Requires root for start/stop/restart; status is
    readable by any user.
    """
    cmd = ["systemctl"] + args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.returncode, result.stdout.strip() + result.stderr.strip()
    except FileNotFoundError:
        return -1, "systemctl not found"
    except subprocess.TimeoutExpired:
        return -1, "systemctl timed out"


@click.group()
def cli():
    """Laya GPU Inference Service CLI."""


@cli.command()
@click.option("--config", "config_path", default=None, help="Path to config JSON")
@click.option(
    "--log-level",
    default="info",
    type=click.Choice(["debug", "info", "warning", "error"]),
)
def serve(config_path: str | None, log_level: str):
    """Start the HTTP server (foreground)."""
    from .server import run_server

    run_server(config_path=config_path, log_level=log_level)


@cli.command()
@click.option("--state", "state_val", required=True, help="JSON string or file path")
@click.option("--questions", "questions_val", required=True, help="JSON string or file path")
@click.option("--json", "as_json", is_flag=True, default=False, help="Raw JSON output")
@click.option("--endpoint", default=None, help="Service endpoint URL")
def query(state_val: str, questions_val: str, as_json: bool, endpoint: str | None):
    """Run inference: POST /system_one."""
    ep = resolve_endpoint(endpoint)
    state = _parse_input(state_val)
    questions = _parse_input(questions_val)
    if not isinstance(questions, dict) or not questions:
        raise click.ClickException("questions must be a non-empty JSON object")

    try:
        result = system_one(ep, state, questions)
    except (ConnectionError, TimeoutError) as e:
        raise click.ClickException(str(e))
    except RuntimeError as e:
        raise click.ClickException(str(e))

    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        _print_answers(result)


@cli.command()
@click.option("--state", "state_val", required=True, help="JSON string or file path")
@click.option("--instructions", required=True, help="Question instructions")
@click.option(
    "--options", "options_val", required=True,
    help='JSON dict {"label":"desc",...} or list ["a","b"]',
)
@click.option("--json", "as_json", is_flag=True, default=False, help="Raw JSON output")
@click.option("--endpoint", default=None, help="Service endpoint URL")
def choice(
    state_val: str, instructions: str, options_val: str, as_json: bool, endpoint: str | None,
):
    """Typed choice: pick one option."""
    from .primitives import laya_choice

    state = _parse_input(state_val)
    options = _parse_input(options_val)
    if isinstance(options, list):
        options = {label: None for label in options}
    if not isinstance(options, dict):
        raise click.ClickException("--options must be a JSON dict or list")

    try:
        result = laya_choice(state, instructions, options, endpoint=endpoint)
    except (ConnectionError, TimeoutError) as e:
        raise click.ClickException(str(e))
    except RuntimeError as e:
        raise click.ClickException(str(e))

    if as_json:
        click.echo(json.dumps(dataclasses.asdict(result), indent=2))
    else:
        click.echo(f"→ {result.choice}")
        for label, prob in result.probabilities.items():
            bar = "█" * int(prob * 30)
            click.echo(f"  {label}: {prob:.4f} {bar}")
        click.echo(f"confidence: {result.confidence:.4f}")
        rs = result.routing_signal
        click.echo(
            f"routing: act={rs.act_probability:.4f} "
            f"escalate={rs.escalate_probability:.4f} (experimental)"
        )


@cli.command()
@click.option("--state", "state_val", required=True, help="JSON string or file path")
@click.option("--instructions", required=True, help="Question instructions")
@click.option(
    "--levels", "levels_val", required=True,
    help='JSON list ["low","med","high"] (ordered, index 0 = lowest)',
)
@click.option("--json", "as_json", is_flag=True, default=False, help="Raw JSON output")
@click.option("--endpoint", default=None, help="Service endpoint URL")
def score(state_val: str, instructions: str, levels_val: str, as_json: bool, endpoint: str | None):
    """Typed score: expected level on an ordered rubric."""
    from .primitives import laya_score

    state = _parse_input(state_val)
    levels = _parse_input(levels_val)
    if not isinstance(levels, list) or len(levels) < 2:
        raise click.ClickException("--levels must be a JSON list with ≥2 items")

    try:
        result = laya_score(state, instructions, levels, endpoint=endpoint)
    except (ConnectionError, TimeoutError) as e:
        raise click.ClickException(str(e))
    except RuntimeError as e:
        raise click.ClickException(str(e))

    if as_json:
        click.echo(json.dumps(dataclasses.asdict(result), indent=2))
    else:
        click.echo(f"score: {result.score:.4f}")
        for lvl, prob in result.probabilities.items():
            bar = "█" * int(prob * 30)
            click.echo(f"  {lvl}: {prob:.4f} {bar}")
        click.echo(f"confidence: {result.confidence:.4f}")
        rs = result.routing_signal
        click.echo(
            f"routing: act={rs.act_probability:.4f} "
            f"escalate={rs.escalate_probability:.4f} (experimental)"
        )


@cli.command()
@click.option("--state", "state_val", required=True, help="JSON string or file path")
@click.option("--statement", required=True, help="Yes/no statement")
@click.option("--true-desc", default=None, help="Description of 'true'")
@click.option("--false-desc", default=None, help="Description of 'false'")
@click.option("--json", "as_json", is_flag=True, default=False, help="Raw JSON output")
@click.option("--endpoint", default=None, help="Service endpoint URL")
def noul(
    state_val: str,
    statement: str,
    true_desc: str | None,
    false_desc: str | None,
    as_json: bool,
    endpoint: str | None,
):
    """Typed noul: P(true) for a yes/no statement."""
    from .primitives import laya_noul

    state = _parse_input(state_val)

    try:
        result = laya_noul(
            state, statement,
            true_desc=true_desc, false_desc=false_desc,
            endpoint=endpoint,
        )
    except (ConnectionError, TimeoutError) as e:
        raise click.ClickException(str(e))
    except RuntimeError as e:
        raise click.ClickException(str(e))

    if as_json:
        click.echo(json.dumps(dataclasses.asdict(result), indent=2))
    else:
        click.echo(f"noul: {result.noul:.4f}")
        rs = result.routing_signal
        click.echo(
            f"routing: act={rs.act_probability:.4f} "
            f"escalate={rs.escalate_probability:.4f} (experimental)"
        )


@cli.command()
@click.option("--state", "state_val", required=True, help="JSON string or file path")
@click.option("--questions", "questions_val", required=True, help="JSON string or file path")
@click.option("--json", "as_json", is_flag=True, default=False, help="Raw JSON output")
@click.option("--endpoint", default=None, help="Service endpoint URL")
def batch(state_val: str, questions_val: str, as_json: bool, endpoint: str | None):
    """Typed batch: many mixed-type questions in one forward pass."""
    from .primitives import laya_batch

    state = _parse_input(state_val)
    questions = _parse_input(questions_val)
    if not isinstance(questions, dict) or not questions:
        raise click.ClickException("--questions must be a non-empty JSON object")

    try:
        result = laya_batch(state, questions, endpoint=endpoint)
    except (ConnectionError, TimeoutError) as e:
        raise click.ClickException(str(e))
    except RuntimeError as e:
        raise click.ClickException(str(e))

    if as_json:
        click.echo(json.dumps(dataclasses.asdict(result), indent=2))
    else:
        from .primitives import ChoiceResult, NoulResult, ScoreResult

        for qid, answer in result.answers.items():
            t = type(answer).__name__
            click.echo(f"\n{qid} ({t}):")
            if isinstance(answer, ChoiceResult):
                click.echo(f"  → {answer.choice}")
                for label, prob in answer.probabilities.items():
                    click.echo(f"    {label}: {prob:.4f}")
            elif isinstance(answer, ScoreResult):
                click.echo(f"  score: {answer.score:.4f}")
            elif isinstance(answer, NoulResult):
                click.echo(f"  noul: {answer.noul:.4f}")
            rs = answer.routing_signal
            click.echo(
                f"  routing: act={rs.act_probability:.4f} "
                f"escalate={rs.escalate_probability:.4f}"
            )
        if result.cached:
            click.echo("\n(cached)")


@cli.command()
@click.option("--json", "as_json", is_flag=True, default=False, help="Raw JSON output")
@click.option("--endpoint", default=None, help="Service endpoint URL")
def health_cmd(as_json: bool, endpoint: str | None):
    """Check service health. Exit: 0=ok, 1=degraded, 2=unreachable."""
    ep = resolve_endpoint(endpoint)
    try:
        data = health(ep)
    except (ConnectionError, TimeoutError):
        click.echo("✗ Laya service unreachable", err=True)
        sys.exit(2)
    except RuntimeError as e:
        click.echo(f"✗ {e}", err=True)
        sys.exit(2)

    if as_json:
        click.echo(json.dumps(data, indent=2))
    else:
        _print_health(data)

    if data.get("status") == "ok":
        sys.exit(0)
    else:
        sys.exit(1)


# Register as "health" (the function name is health_cmd to avoid clash with client.health)
health_cmd.name = "health"


@cli.command()
@click.option("--json", "as_json", is_flag=True, default=False, help="Raw JSON output")
@click.option("--endpoint", default=None, help="Service endpoint URL")
def stats_cmd(as_json: bool, endpoint: str | None):
    """Show service statistics."""
    ep = resolve_endpoint(endpoint)
    try:
        data = stats(ep)
    except (ConnectionError, TimeoutError) as e:
        raise click.ClickException(str(e))
    except RuntimeError as e:
        raise click.ClickException(str(e))

    if as_json:
        click.echo(json.dumps(data, indent=2))
    else:
        _print_stats(data)


# Register as "stats"
stats_cmd.name = "stats"


@cli.command()
@click.option("--endpoint", default=None, help="Service endpoint URL")
def unload_cmd(endpoint: str | None):
    """Unload the model from GPU memory."""
    ep = resolve_endpoint(endpoint)
    try:
        data = unload(ep)
    except (ConnectionError, TimeoutError) as e:
        raise click.ClickException(str(e))
    except RuntimeError as e:
        raise click.ClickException(str(e))

    if data.get("status") == "unloaded":
        click.echo("Model unloaded from GPU.")
    else:
        click.echo(f"Model was already unloaded ({data.get('status')})")


unload_cmd.name = "unload"


@cli.command()
def version():
    """Print the laya-service version."""
    click.echo(f"laya-service {__version__}")


@cli.command("start")
def start():
    """Start the laya-service (system unit; requires root)."""
    rc, output = _run_systemctl(["start", "laya-service"])
    if rc != 0:
        click.echo(f"✗ Failed to start: {output}", err=True)
        sys.exit(1)
    click.echo("✓ laya-service started")


@cli.command("stop")
def stop():
    """Stop the laya-service (system unit; requires root)."""
    rc, output = _run_systemctl(["stop", "laya-service"])
    if rc != 0:
        click.echo(f"✗ Failed to stop: {output}", err=True)
        sys.exit(1)
    click.echo("✓ laya-service stopped")


@cli.command("restart")
def restart():
    """Restart the laya-service (system unit; requires root)."""
    rc, output = _run_systemctl(["restart", "laya-service"])
    if rc != 0:
        click.echo(f"✗ Failed to restart: {output}", err=True)
        sys.exit(1)
    click.echo("✓ laya-service restarted")


@cli.command()
@click.option("--json", "as_json", is_flag=True, default=False, help="Stable JSON output")
@click.option("--endpoint", default=None, help="Service endpoint URL")
def status(as_json: bool, endpoint: str | None):
    """Combined systemd + health status.

    Exit codes:
      0 = active/healthy
      1 = running but degraded (model unloaded)
      2 = process running but API unreachable, or state unknown
      3 = service stopped
    """
    ep = resolve_endpoint(endpoint)

    # Get systemd state
    rc, output = _run_systemctl(["is-active", "laya-service"])
    if rc == 0 and "active" in output:
        systemd_state = "active"
    elif "inactive" in output:
        systemd_state = "inactive"
    elif "unknown" in output or rc != 0:
        systemd_state = "unknown"
    else:
        systemd_state = output.strip() or "unknown"

    # Get health
    health_data = None
    health_status = "unreachable"
    if systemd_state == "active":
        try:
            health_data = health(ep)
            health_status = health_data.get("status", "unknown")
        except (ConnectionError, TimeoutError):
            health_data = None
            health_status = "unreachable"
        except RuntimeError:
            health_data = None
            health_status = "unreachable"

    # Build output
    result = {
        "systemd": systemd_state,
        "health": health_data,
        "endpoint": ep,
    }

    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        if systemd_state == "inactive":
            click.echo("○ laya-service: stopped")
        elif systemd_state == "active" and health_status == "ok":
            click.echo("● laya-service: healthy")
            if health_data:
                click.echo(f"  Model: loaded, uptime {health_data.get('uptime_s', 0)}s")
        elif systemd_state == "active" and health_status == "degraded":
            click.echo("○ laya-service: running but model unloaded (degraded)")
        elif systemd_state == "active" and health_status == "unreachable":
            click.echo("✗ laya-service: process running but API not responding")
        else:
            click.echo(f"? laya-service: state={systemd_state}, health={health_status}")

    # Exit code
    if systemd_state == "inactive":
        sys.exit(3)
    elif systemd_state == "active" and health_status == "ok":
        sys.exit(0)
    elif systemd_state == "active" and health_status == "degraded":
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    cli()
