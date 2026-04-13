#!/usr/bin/env bash
# Measure peak memory for each pytest test case.
# Usage: ./measure_test_memory.sh tests-list.txt [output.csv]
#
# Each test is run in a separate pytest invocation so that memory is isolated.
# A background polling loop samples macOS `footprint` every 0.5s to track the
# peak logical memory footprint (matches Activity Monitor's "Memory" column).
# Also records peak RSS from /usr/bin/time -l for comparison.

set -euo pipefail

INPUT="${1:?Usage: $0 tests-list.txt [output.csv]}"
OUTPUT="${2:-test_memory.csv}"
POLL_INTERVAL=0.5

echo "test,peak_rss_bytes,peak_footprint_bytes,duration_s,status" > "$OUTPUT"

total=$(wc -l < "$INPUT" | tr -d ' ')
i=0

while IFS= read -r test_id; do
    i=$((i + 1))

    # Skip blank lines
    [[ -z "$test_id" ]] && continue

    printf "[%d/%d] %s ... " "$i" "$total" "$test_id"

    # Temp file for the polling loop to write peak footprint
    peak_file=$(mktemp)
    echo 0 > "$peak_file"

    # Run the test in the background, capturing all output
    /usr/bin/time -l uv run pytest "$test_id" --no-header -q --timeout=300 \
        > "$peak_file.pytest" 2> "$peak_file.time" &
    test_pid=$!

    # Find the actual python child process and poll its footprint.
    # We poll the pytest (python) process, not `uv run` or `time`.
    max_footprint=0
    while kill -0 "$test_pid" 2>/dev/null; do
        # Find python child processes of the test
        for pid in $(pgrep -P "$test_pid" python 2>/dev/null || true) $(pgrep -P "$(pgrep -P "$test_pid" 2>/dev/null | head -1)" python 2>/dev/null || true); do
            # Get footprint header line, e.g. "python3 [PID]: 64-bit    Footprint: 1234567 B ..."
            fp=$(footprint -p "$pid" -f bytes 2>/dev/null \
                | head -2 | grep 'Footprint:' | sed 's/.*Footprint: \([0-9]*\) B.*/\1/' || true)
            if [[ -n "$fp" ]] && (( fp > max_footprint )); then
                max_footprint=$fp
            fi
        done
        sleep "$POLL_INTERVAL"
    done

    # Wait for the test to finish and capture exit code
    wait "$test_pid" 2>/dev/null || true

    # Parse peak RSS from /usr/bin/time output (written to stderr)
    time_output=$(cat "$peak_file.time")
    peak_rss=$(echo "$time_output" | grep 'maximum resident set size' | awk '{print $1}')
    duration=$(echo "$time_output" | grep 'real' | awk '{print $1}')

    # Determine pass/fail from pytest stdout
    pytest_output=$(cat "$peak_file.pytest")
    if echo "$pytest_output" | grep -q 'passed'; then
        status="passed"
    elif echo "$pytest_output" | grep -q 'failed'; then
        status="failed"
    elif echo "$pytest_output" | grep -q 'error'; then
        status="error"
    else
        status="unknown"
    fi

    peak_rss="${peak_rss:-0}"
    duration="${duration:-0}"
    rss_mb=$(echo "scale=0; ${peak_rss} / 1048576" | bc)
    foot_mb=$(echo "scale=0; ${max_footprint} / 1048576" | bc)

    printf "rss=%s MB  footprint=%s MB  %ss  %s\n" "$rss_mb" "$foot_mb" "$duration" "$status"
    echo "${test_id},${peak_rss},${max_footprint},${duration},${status}" >> "$OUTPUT"

    # Clean up temp files
    rm -f "$peak_file" "$peak_file.time" "$peak_file.pytest"

done < "$INPUT"

echo ""
echo "Done. Results written to $OUTPUT"
