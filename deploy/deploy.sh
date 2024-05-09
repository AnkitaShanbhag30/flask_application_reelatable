#! /bin/sh
docker build --platform linux/amd64 -t gcr.io/reelatable-420506/docker-flask-reelatable:latest .
docker push gcr.io/reelatable-420506/docker-flask-reelatable:latest
cd deploy
kubectl apply -f flask-reelatable-deployment.yaml
kubectl apply -f flask-reelatable-service.yaml
cd ..
kubectl rollout restart deployment flask-reelatable
kubectl get pods