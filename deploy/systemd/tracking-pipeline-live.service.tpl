[Unit]
Description=Tracking Pipeline Live Run
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=@SERVICE_USER@
WorkingDirectory=@REPO_ROOT@
EnvironmentFile=-@ENV_FILE@
Environment=PYTHONUNBUFFERED=1
ExecStart=@REPO_ROOT@/scripts/run_live_service.sh
Restart=on-failure
RestartSec=5
TimeoutStopSec=45
KillSignal=SIGTERM

[Install]
WantedBy=multi-user.target
