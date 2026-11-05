Full upstream Volta deployment snapshot — 2026-11-05

The reference host now runs localhost/ninfer-v100:upstream-f4e186aa via the user
Quadlet ninfer.service (container ninfer-v100). The inference revision remains
f4e186aa9365357e69ef26f052aef6d5cf6b4674; later documentation commits do not
change the validated binaries. deployment.json records the original rollout;
context232-deployment.json records the current 232000-token configuration and checks.

ninfer.container and entrypoint-volta.sh are copies of the installed configuration.
They target this host's absolute model, driver, log, and release paths; adapt them
before use on another machine. The image and weights are local host artifacts,
not published container images or binaries included in Git.

The installed launcher is /home/dr_cain/projects/ninfer/entrypoint-volta.sh.
The retained release /home/dr_cain/projects/ninfer/out/upstream-v100-f4e186aa
contains the tested seccomp-uring.json policy used by the Quadlet, packaged image
metadata, and qualification report. The model is the losslessly packed artifact
at /home/dr_cain/projects/ninfer/out/models/qwen3.8-27b-volta-v3.ninfer.
Do not remove these files while this deployment is active.

The deployment preserves port 8080, alias qwen3.8-27b, max context 232000,
concurrency 2, INT8 KV, prefill chunk 2048, MTP4, CUDA graphs, and vision. The new
engine uses the qualified 9221505024-byte host cache and optimized proposal head.
The pending timeout remains 600s. Kernel io_uring policy, the tested container
seccomp profile, and unlimited memlock allow the runtime's I/O operations.

Post-switch validation passed 16/16 API regression checks and 8/8 concurrent
arithmetic checks. Executable and Ops library hashes match the qualified release.
That original rollout advertised 200000 tokens; the current service advertises 232000. No service restarts or process swap were
observed during these checks. Earlier numerical, sanitizer, performance, and
long-context results are documented in ../../docs/validation/2026-11-05/REPORT.txt.

The 232K configuration completed the synthetic test documented in
../../docs/validation/2026-11-05/context232/REPORT.txt. It qualifies one long
conversation with smaller competing requests, not two full 232K conversations.
Concurrency remains 2 and the host-cache budget is unchanged. The embedded model
template is in use; high is rejected, while xhigh, medium and low are native.
Hermes retains its separate 200000-token client cap.
To restore the previous 200K full-port configuration on this host, run:
  /home/dr_cain/projects/ninfer/deployments/context232-live-20261105/rollback.sh

Logs: podman logs ninfer-v100
Structured log: /home/dr_cain/projects/ninfer/logs/requests-upstream-f4e186aa.jsonl
Daily startup: /home/dr_cain/projects/ninfer/start.sh

The reliability image localhost/ninfer-v100:sm70-bc3f1be2 is retained for rollback.
The host snapshot /home/dr_cain/projects/ninfer/deployments/upstream-f4e186aa-20261105
contains the previous Quadlet and launcher, validation output, and rollback.sh.
Run that script to stop the full port, restore the reliability configuration,
reload systemd, and restart. Verify /health after model initialization.
