### qnas/docker

Tools for distributed network training

```

    - run-master.sh -- Pushes jobs to a queue
    - run-worker.sh -- Runs workers that read model specs from queue and train
    
```

Need to create a `credentials.sh` file that looks like:

```
    export QNAS_HOST=... redis host
    export QNAS_PORT=... redis port
    export QNAS_PASSWORD=... redis password
```