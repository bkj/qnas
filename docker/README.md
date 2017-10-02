### qnas/docker

Tools for distributed network training

```

    - master.py -- Pushes jobs to a queue
    - worker.py -- Runs jobs
    
```

Need to create a `credentials.sh` file that looks like:

```
    export QNAS_HOST=... redis host
    export QNAS_PORT=... redis port
    export QNAS_PASSWORD=... redis password
```