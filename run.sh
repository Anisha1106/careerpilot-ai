#!/bin/sh
set -eu

app_url="http://localhost:5173"

docker compose up --build --wait --wait-timeout 120 -d

if command -v open >/dev/null 2>&1; then
    open "$app_url"
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$app_url" >/dev/null 2>&1 &
else
    printf 'CareerPilot AI is running at %s\n' "$app_url"
fi