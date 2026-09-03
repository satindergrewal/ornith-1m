#!/usr/bin/env bash
# Install the 15-minute watchlist cron entry for Satinder.
# Idempotent: preserves existing crontab entries, replaces our entry in place.
# 2026-09-03: live dir moved to ~/watchlist — cron processes lose TCC access
# to ~/Documents ("Operation not permitted"), home-root paths are exempt.
# Repo tools/watchlist/ stays the canonical source; sync copies OUT before
# running and deltas/state BACK when convenient.
ENTRY='*/15 * * * * cd /Users/satinder/watchlist && /usr/bin/python3 check.py >> run.log 2>&1'
MARK='cd /Users/satinder/watchlist && /usr/bin/python3 check.py'
OLD_MARK='tools/watchlist && /usr/bin/python3 check.py'
CURRENT="$(crontab -l 2>/dev/null | grep -vF "$MARK" | grep -vF "$OLD_MARK" || true)"
if [ -n "$CURRENT" ]; then
    printf '%s\n%s\n' "$CURRENT" "$ENTRY" | crontab -
else
    printf '%s\n' "$ENTRY" | crontab -
fi
echo "INSTALLED --- crontab now:"
crontab -l
