V100 Podman deployment snapshot

This directory records the reliability-fixes deployment validated on 2026-11-04.
The image is localhost/ninfer-v100:sm70-bc3f1be2, built from source commit
bc3f1be27817d4fdb9db8f5c88ffc43555569163 on branch
backport/upstream-reliability-20261103. It contains the working V100 engine plus
selected upstream reliability fixes. The separate port/upstream-v100-20261103
branch subsequently completed full-engine V100 qualification on 2026-11-05;
see its README for results and its different configuration requirements.
This snapshot continues to describe the deployed reliability build.

ninfer.container is the deployed user Quadlet. entrypoint.sh maps its environment
to server options. The model and driver-library paths match the validated host;
adapt these paths for another machine. The service mounts the launcher from
/home/dr_cain/projects/ninfer/entrypoint.sh, so that file must be installed there
before starting the Quadlet.

The tested image is local to the deployment host, not published to a container
registry. Neither model weights nor the server executable are stored in Git.
The validated package is retained on the deployment host at:
  /home/dr_cain/projects/ninfer/releases/ninfer-v100-bc3f1be2.tar.gz
Its BUILDINFO.txt, Containerfile.builder and build-in-container.sh record the
CUDA 12.8.1 / Ubuntu 24.04 / sm70 build recipe and pinned CUTLASS revision.

On the validated host, installation uses:
  install -m 0644 ninfer.container ~/.config/containers/systemd/ninfer.container
  systemctl --user daemon-reload
  systemctl --user restart ninfer.service

The previous image localhost/ninfer-v100:sm70-v2 is retained for rollback. Its
previous Quadlet is saved on the deployment host at:
  /home/dr_cain/projects/ninfer/deployments/reliability-20261104/ninfer.container.previous

validation/2026-11-04 contains the benchmark report, aggregate results and per-run
measurements. Full request logs and the benchmark harness remain on the host at:
  /home/dr_cain/projects/ninfer/benchmarks/2026-11-04/
The report describes both versions and the test limits. After benchmarking, the
reliability-fixes image was deployed and passed health, chat, streaming, concurrent
request, thinking-token accounting and executable/image identity checks.
