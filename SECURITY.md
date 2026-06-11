# Security Policy

## Reporting a Vulnerability

Report vulnerabilities privately via GitHub's "Report a vulnerability" button
on the Security tab of this repository (GitHub Private Vulnerability Reporting).
Please do not open public issues for security problems.

You can expect an acknowledgement within a week. Fixes ship as a patch release
with credit unless you prefer otherwise.

## Scope and Known Trade-offs

- **All processing is local.** Audio, transcripts, and settings never leave the
  machine. The only network traffic is the one-time model download from
  Hugging Face on first use of a model size.
- **Model downloads** are fetched by faster-whisper from the upstream
  `Systran/faster-whisper-*` repositories over HTTPS and cached locally.
- **Logs** are written to the local app-data directory and deliberately record
  only file basenames, never full paths. Delete the log file at any time if
  you transcribe sensitive material.
- **Dependencies** are pinned (requirements.txt, pyproject.toml) with the full
  transitive closure locked in uv.lock; Dependabot watches for advisories.

## Supported Versions

Only the latest release receives security fixes.
