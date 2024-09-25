
# Installation & Setup

## 1. Create Virtual Environment & Activate

### Windows
- Create environment:  
  ```bash
  virtualenv env
  ```
- Activate environment:  
  ```bash
  venv\Scripts\activate
  ```

### Linux
- Create environment:  
  ```bash
  python -m venv env
  ```
- Activate environment:  
  ```bash
  source venv/bin/activate
  ```

## 2. Move to Project Directory
Ensure you're in the project directory before proceeding.

## 3. Install Dependencies
Run the following command to install all required dependencies:  
```bash
pip install -r requirements.txt
```

## 4. Configure Gunicorn & Nginx (Production)
Configure `gunicorn` and `nginx` for your production environment. Refer to the documentation for detailed instructions.

## 5. Start Server
Run the application using your preferred method (e.g., using `gunicorn`):  
```bash
gunicorn --config gunicorn_config.py BetacraftCodeAnalyst.wsgi
```


## For User Instructions: https://docs.google.com/document/d/1a8b4UVeq3qRsgXUvfWxHNuHsoBVAOYrKBZLnaL_Q9R4/edit?usp=sharing
