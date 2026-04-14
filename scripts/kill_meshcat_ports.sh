#!/bin/bash

# Define the ports to kill
PORTS=(6000 7000)

echo "Starting to clear Meshcat server ports..."

for port in "${PORTS[@]}"; do
    echo "Checking port $port..."
    
    # Using lsof to find the PID
    PIDS=$(lsof -t -i:$port)
    
    if [ -z "$PIDS" ]; then
        echo "Port $port is free."
    else
        echo "Found process(es) listening on $port: $PIDS"
        
        # Kill the process
        for pid in $PIDS; do
            echo "Killing PID $pid..."
            kill -9 $pid
            if [ $? -eq 0 ]; then
                echo "Successfully killed PID $pid."
            else
                echo "Failed to kill PID $pid."
            fi
        done
    fi
done

echo "Done clearing ports."
