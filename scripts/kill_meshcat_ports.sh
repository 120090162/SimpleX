#!/bin/bash

# Define the port ranges to clear
PORT_RANGES=("6000 6010" "7000 7010")

echo "Starting to clear Meshcat server ports..."

for range in "${PORT_RANGES[@]}"; do
    read -r start_port end_port <<< "$range"
    for ((port = start_port; port <= end_port; port++)); do
        echo "Checking port $port..."

        # Use lsof to find PIDs listening on this port
        PIDS=$(lsof -t -i:"$port")

        if [ -z "$PIDS" ]; then
            echo "Port $port is free."
        else
            echo "Found process(es) listening on $port: $PIDS"

            for pid in $PIDS; do
                if kill -0 "$pid" 2>/dev/null; then
                    echo "Killing PID $pid..."
                    kill -9 "$pid"
                    if [ $? -eq 0 ]; then
                        echo "Successfully killed PID $pid."
                    else
                        echo "Failed to kill PID $pid."
                    fi
                fi
            done
        fi
    done
done

echo "Done clearing ports."
