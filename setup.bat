python -m venv venv
call venv\Scripts\activate
pip install fastapi uvicorn sentence-transformers scikit-learn numpy
uvicorn main:app --reload