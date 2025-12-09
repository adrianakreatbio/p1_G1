source ../py_env/bin/activate
pip install -r ../requirements.txt
uvicorn local_server:app --host 127.0.0.1 --port 8000

#   WORKFLOW
#   open server connected to app in local_server -> (flutter run -d windows) open app -> (press measure button) call app



