# Retrieval Agents
## Overview

## Quick Start

### Prerequisites

- **Python 3.8+** (recommended: Python 3.9 or 3.10)
- **Jupyter Notebook** or **JupyterLab**
- **Git** for cloning the repository

### Installation

1. **Clone the repository**
   ```bash
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # Using venv
   python -m venv archaas_env
   
   # Activate virtual environment
   # On Windows:
   archaas_env\Scripts\activate
   # On macOS/Linux:
   source archaas_env/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

   Or install packages individually:
   ```bash
   pip install openai anthropic python-dotenv jupyter
   pip install agents pydantic requests wikipedia-api
   pip install boto3 botocore IPython
   ```

### Configuration

1. **Create environment file**
   ```bash
   # Create .env file in the project root
   touch .env
   ```

2. **Add your API keys** to the `.env` file:
   ```env
   # Required API Keys
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   
   # Optional: AWS Bedrock (if you have access)
   AWS_ACCESS_KEY_ID=your_aws_access_key
   AWS_SECRET_ACCESS_KEY=your_aws_secret_key
   AWS_DEFAULT_REGION=us-east-1
   ```

### Running

1. **Start Jupyter**
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```