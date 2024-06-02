In this section, model training is done on a local machine, then deployed using CloudBuild based on the principle of CI/CD. With every code change, CloudBuild deploys the changed code in the form of an updated docker image on Google Container Registry to Google CloudRun, which will redeploy the model API (the model API runs on Flask).

The code can be found at: https://github.com/alanchn31/stroke-prediction-cloudbuild-cicd
