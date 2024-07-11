# ML distributed: Handling Traffic with Celery and/or Ray.io

## Add a fake subheader

Edited on Github

## Introduction

In today's world, machine learning (ML) models and implementation use by multitude of people. It is common to find data scientist writing their projects in Jupyter Notebooks or creating apps to expose their projects or services to a group of people or to be public. In this latter case, many times you can end up releasing a service that you didn't expect to become popular, or at least that a lot of people end up using. You probably need to deploy your system in a cluster on some cloud provider and may require specialized hardware for it to function. Many AI models consume a lot of RAM and GPU and can easily overwhelm your memory when running simultaneous processes. Or perhaps you have a single server with enough resources to run one process at a time. In those cases, you have to be prepared and design your service to handle high traffic and ensure smooth operation. To address this challenge, implementing a queue system becomes essential. 

In this blog post, we will explore the use of Celery, a useful distributed task queue system, to efficiently manage high traffic in ML production environments.We are going to create a project that generates images with AI through a prompt. We will start by using the default task architecture provided by Celery. This architecture allows us to define tasks as functions and execute them asynchronously. I will show how to create tasks for our AI implementation and execute them using Celery. However, we will also explore the problem of downloading or loading models every time a task is executed using this default setup. To overcome this limitation, we will delve into Celery's custom task implementation and highlight its benefits. By customizing our tasks, we can optimize the model loading process and enhance the overall performance of our ML application. We will show how to create custom tasks that load the ML models once and reuse them for subsequent tasks. This approach significantly reduces the overhead of model loading and improves the efficiency of our ML application.

The flow we are going to follow is as follows:

* I will show you the different components and the implementation of each one for the project, but without the implementation of Celery or Ray.io. This is so you can see how the project would work without a distribution tool why is this important - to scale system and have many users at one time.
* I will show you how Celery would be integrated, along with some strategies to consider. Finally, we will observe its functioning.
* I will show you other alternatives that could be useful to you.

## Problem Statement: Creating an AI Application for Image Generation

The app  we want to implement consists in an AI application that allows us to generate images from a prompt using stable diffusion models. The goal is to develop an interface where clients can input a prompt, and the application will process the request, execute the AI pipeline, and return the generated image as a result. The process involves leveraging stable diffusion models, such as text2img and img2img, to generate images based on the provided prompt. reconize

To achieve this, we will create a user-friendly interface using frameworks like FastAPI for the backend and StreamLit for the frontend. The backend will focus on the ML implementation while the frontend will be the interface through which users will interact. FastAPI Python web framework that enables building high-performance, production-ready APIs with minimal coding. It speeds up development thanks to automatic validation and documentation driven by Python type annotations. Key benefits are blazing-fast performance comparable to NodeJS/Go, rapid coding/iteration, robustness for complex applications, as well as easy scalability to handle sizable traffic.   On the other hand, StreamLit is a user-friendly library that allows us to quickly create interactive data visualizations and applications. If you want something more robust for the frontend, you would need to use a JavaScript framework such as React or NextJS. However, for rapid prototyping, we will use Streamlit.

The workflow will involve the client entering a prompt through the interface. The frontend will send a request to the backend, triggering the AI pipeline to process the prompt and generate the corresponding image. The generated image will then be returned to the client as the result of the request.

## Implementing AI applications with FastAPI and StreamLit [No Celery]

<small>The code implemented for this section can be found at: https://github.com/curibe/ml-production-celery/tree/front-back-no-celery</small>

We are going to use Streamlit to build the interface with which the user will interact with an API built in FastAPI that will contain the image generation pipeline. Typically, when using Streamlit, a common trend is to create the interface directly integrated with the code that implements the ML pipeline, as is commonly done with SSR in Python, such as Flask or Django. However, the idea is to implement decoupled structures so that the project is easy to maintain. For this reason, we will create two separate services: "client," which will contain the implementation in Streamlit, and "server," which will contain the implementation with FastAPI. This would allow us, for example, to change the interface made in Streamlit to one made with a JavaScript framework like ReactJS or NextJS, Angular, VueJS, etc., in case we want to implement a more robust and scalable interface without altering the API.

![image](https://hackmd.io/_uploads/H1LQdzswC.png)

Let's quickly see how the image generation application is structured.


### Server with FastAPI

To implement the server or API, we will make use of:
* FastAPI for building the API
* Poetry for managing dependencies
* Docker for encapsulating and running the API in an isolated container
* Docker Compose for managing and running various services associated with the AI application

One of the advantages of using FastAPI is that it allows you to prototype and set up an application quickly and efficiently. We could create the API with a simple structure; however, we will implement it as a hybrid between a layered and hexagonal architecture (ports and adapters). We do this to create an application that is scalable, testable, and easy to maintain. By creating interfaces and using dependency injection, it enables us to add features more rapidly. With this in mind, the server's structure will be as follows (excluding the implementation of Celery for now)


```text
.
├──  README.md
├──  docker-compose.yaml
└──  server
    ├──  Dockerfile
    ├──  app
    │   ├──  __init__.py
    │   ├──  app.py
    │   ├──  config
    │   │   ├──  __init__.py
    │   │   ├──  logger.py
    │   │   └──  logging-conf.yaml
    │   ├──  integrations
    │   │   ├──  __init__.py
    │   │   ├──  genai_generator.py
    │   │   └──  generation_interface.py
    │   ├──  models
    │   │   ├──  __init__.py
    │   │   └──  schemas.py
    │   ├──  operators
    │   │   ├──  __init__.py
    │   │   └──  stable_diffusion.py
    │   ├──  services
    │   │   ├──  __init__.py
    │   │   └──  generation_service.py
    │   └──  utils
    │       ├──  __init__.py
    │       └──  images.py
    ├──  main.py
    ├──  poetry.lock
    └──  pyproject.toml

```

Let's see what each of these elements is:

 * **utils**. It contains all the functionalities or utilities that are cross-cutting in the application
 * **app.py**. It contains everything related to the creation of the API as well as the available endpoints.
    ```python
    from io import BytesIO
    from pathlib import Path

    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse

    from app.config import get_settings
    from app.config.logger import InitLogger
    from app.integrations.genai_generator import GenerativeAIGenerator
    from app.models.schemas import GenRequest
    from app.services.generation_service import GenerationService
    from app.utils.images import from_image_to_bytes

    app = FastAPI()

    # Create service to generate an image with Diffusion models
    # We inject the image generator integration dependency in the service
    generator_service = GenerationService(generator=GenerativeAIGenerator())

    # Load config to Logging system
    config_path = Path("app/config").absolute() / "logging-conf.yaml"
    logger = InitLogger.create_logger(config_path)

    # Load settings
    settings = get_settings()

    ALLOWED_ORIGINS = settings.allowed_origins.split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


    @app.get('/')
    async def root():
        return {"message": "Hello World"}


    @app.post('/generate')
    async def generate(request: GenRequest):
        # Call the service to generate the images according to the request params
        images = generator_service.generate_images_with_text2img(request=request)
        img_bytes = from_image_to_bytes(images[0])
        return StreamingResponse(BytesIO(img_bytes), media_type="image/png")

    ```
    
    In this file, we import some of the other components such as settings, the image generator service, the image generator, and some Pydantic schemas. It also loads the configuration that defines the format of the logs created with loguru (a logging library).

* **Config**. It contains the configuration files for the system. Here you will find functionalities to define and load the logging system configuration. In the `__init__.py` file, project environment variables are defined.

    ```python
    from functools import lru_cache

    from pydantic_settings import BaseSettings


    class Settings(BaseSettings):
        allowed_origins: str = "*"
        default_scheduler: str = "DDPMScheduler"
        default_pipeline: str = "StableDiffusionXLPipeline"
        default_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
        max_num_images: int = 4


    @lru_cache()
    def get_settings() -> Settings:
        settings = Settings()
        return settings

    ```

    In the Settings class, the environment variables required by the application are defined. These variables have default values, but you can change them directly in this file or create a .env file with these variables (recommended way).
    
* **Models**. It contains the definition of entities and schemas used throughout the entire project.
    ```python
    from pydantic import BaseModel

    from app.config import get_settings

    settings = get_settings()


    class GenRequest(BaseModel):
        prompt: str = ("logo of The Witcher Portrait for a spiritual Premium Instagram channel,artwork, fujicolor, "
                       "trending on artstation, faded forest background")
        negative_prompt: str = "bad, low res, ugly, deformed"
        width: int = 768
        height: int = 768
        num_inference_steps: int = 60
        guidance_scale: float = 5
        num_images_per_prompt: int = 1
        generator: int = 88300
        pipeline: str = settings.default_pipeline
        scheduler: str = settings.default_scheduler
        modelid: str = settings.default_model


    class GenResponse(BaseModel):
        image: str

    ```

    `GenRequest`  is a Pydantic model that defines the parameters required by the API and, consequently, by the diffusion model, to generate images. These models allow us to validate the parameters received by the API. You will also notice that this model has default values defined for each parameter. This allows us to send a request to the API without the need to include these parameters in the request, providing a default behavior for the API. If you want to understand the meaning of each parameter, you can refer to sources such as https://diffusion-news.org/stable-diffusion-settings-parameters. 

* **Services**. This layer is commonly used to implement all the necessary business logic for the application. In our case, as it is an application focused on a specific task, the implementation of this layer may not contain elaborate logic. However, it is important because it allows us to decouple and connect the API with integrations, repositories, etc.

    ```python
    from app.integrations.generation_interface import GenerationInterface
    from app.models.schemas import GenRequest


    class GenerationService:
        def __init__(self, generator: GenerationInterface):
            self.generation = generator

        def generate_images_with_text2img(self, request: GenRequest):
            generation = self.generation.generate_image_with_text2img(request=request)
            return generation
    ```
    
    The service receives a generator that is responsible for integrating AI models into our application. This generator is injected as a dependency and is defined as an interface, specifying the general functionalities and operations it should have. This provides us with the versatility to create different generators and integrate them into our API easily.
    

* **Integrations**. This component, as its name indicates, is used to integrate external services or other internal components. In our case, we use it to integrate the component responsible for generating images using AI.

    <small><em><b>interface:</b></em></small>
    ```python
    from abc import ABC, abstractmethod


    class GenerationInterface(ABC):
        @staticmethod
        @abstractmethod
        def generate_image_with_text2img(*, request) -> str:
            raise NotImplementedError("generate_image_with_text2img method is not implemented")
    ```
    
    <small><em><b>generator:</b></em></small>
    ```python
    from app.integrations.generation_interface import GenerationInterface
    from app.models.schemas import GenRequest
    from app.operators.stable_diffusion import StableDiffusionText2Image


    class GenerativeAIGenerator(GenerationInterface):
        @staticmethod
        def generate_image_with_text2img(*, request: GenRequest) -> str:
            generator = StableDiffusionText2Image()
            generation = generator.generate_images(request=request)
            return generation
    ```
    
    As you can note, `GenerativeAIGenerator` implements the Interface which define main functionalities a Generator must have. In this case, there is only one functionality: generate an image by a prompt (text to image). But we can include more features to our app like generating images from a base image and a prompt (image to image), upscaling, removing backgrounds, etc. just adding this new functionality to the interface and generator. Now, `GenerativeAIGenerator` does not implement all the generative pipeline but import a module for that. Maybe you are thinking: why not just implement this pipeline here instead of creating a new module? Well, Imagine that you want to create a new Generator and use the same pipeline. This allows us to avoid repeated code and also to integrate new generators easily
    
    
### AI Implementation: Stable Diffusion with text2img 

As part of our ML application, we will showcase an example of Stable Diffusion with text2img This implementation will serve as a starting point for our code. We will use text2img to generate images based on input text. The module stable_diffusion.py contains a pipeline that implements text to image, generating images through a prompt. Now, let's take a look at the implementation of this operator:

```python
import importlib

import torch
from loguru import logger

from app.config import get_settings
from app.models.schemas import GenRequest

settings = get_settings()


class StableDiffusionText2Image:
    def __init__(
            self,
            model_name: str = settings.default_model,
            scheduler: str = "PNDMScheduler",
            pipeline_name: str = "StableDiffusionXLPipeline",
    ):
        self.model_name = model_name
        self.scheduler = scheduler

        self.dtype = torch.float16

        if torch.cuda.is_available():
            self.generator_device = "cuda"
            self.device = "cuda"
        else:
            self.generator_device = "cpu"
            self.device = "cpu"

        self._module_import = importlib.import_module("diffusers")
        self._pipeline = getattr(self._module_import, pipeline_name)

        self.pipe = self._pipeline.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
        )
        self.model = self.pipe.to(self.device)

        # Set param optimization
        self.model.enable_attention_slicing()

        # import scheduler
        scheduler = getattr(self._module_import, self.scheduler)
        self.model.scheduler = scheduler.from_config(self.model.scheduler.config)

    def generate_images(self, **kwargs):
        logger.info("generating image in Text2Image pipeline")
        request: GenRequest = kwargs.get("request")
        generator = torch.Generator(self.generator_device).manual_seed(request.generator)

        images = self.model(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_images_per_prompt=request.num_images_per_prompt,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            generator=generator,
            width=request.width,
            height=request.height,
        ).images

        if len(images) == 0:
            logger.info("Unable to generate text2img images")
            return None

        logger.info("text2img completed successfully")
        return images

```

In the `__init__` function of this pipeline, several initialization operations are performed. We make use of the Hugging Face diffusers library to load the models. You'll notice that the Hugging Face pipeline is passed as a parameter and then imported dynamically. This allows us to load and use this class for more than one pipeline. For example, we could use StableDiffusionPipeline, DiffusionPipeline, StableDiffusionControlNetPipeline, StableDiffusionXLImg2ImgPipeline (check [here](https://huggingface.co/docs/diffusers/api/pipelines/overview) for more pipelines). The same applies to schedulers or samplers; we can use various ones (check [here](https://huggingface.co/docs/diffusers/api/schedulers/overview) for more schedulers). Unlike these mentioned parameters, which are needed when instantiating the Hugging Face pipeline, the `generate_images` function receives more parameters that can be tunable during execution. These parameters allow us to control certain aspects of the resulting image and, therefore, adjust the outcome as much as we want.


### Client with Streamlit
The client is an isolated component in Python that will make use of:
* Streamlit to build the interface.
* It uses a native dependency manager: pip with requirements.
* Docker to run the interface in an isolated component.
* Docker Compose to execute the service.

Being a decoupled component separated from the API or server implementation, the communication between the client and the ML pipeline will be via HTTP, similar to how it is done between modern JavaScript frameworks and a backend. The structure of the client is simple as it does not require many dependencies:

```text
.
├── README.md
├── client
│   ├── Dockerfile
│   ├── client.py
│   ├── config.py
│   ├── requirements.txt
│   └── utils.py
├── docker-compose.yaml
```

Let's delve into some of these files in detail:

* **config.py**. It contains some useful parameters for the application, such as the diffusion models to use, the schedulers, and it also includes a Pydantic schema defined in the server.

    ```python
    from pydantic import BaseModel

    model_map = {
        "Stable Diffusion XL": "stabilityai/stable-diffusion-xl-base-1.0",
        "Stable Diffusion 2": "stabilityai/stable-diffusion-2"
    }

    pipeline_map = {
        "stabilityai/stable-diffusion-xl-base-1.0": "StableDiffusionXLPipeline",
        "stabilityai/stable-diffusion-2": "StableDiffusionPipeline"
    }

    scheduler_map = {
        'Euler': 'EulerDiscreteScheduler',
        'Euler a': 'EulerAncestralDiscreteScheduler',
        'DPM2': 'KDPM2DiscreteScheduler',
        'LMS': 'LMSDiscreteScheduler',
        'Heun': 'HeunDiscreteScheduler',
        'DPM2 a': 'KDPM2AncestralDiscreteScheduler',
        'DDPM': 'DDPMScheduler',
        'DPM++ 2M': 'DPMSolverMultistepScheduler',
        'DDIM': 'DDIMScheduler',
        'DPM': 'DPMSolverSinglestepScheduler',
        'DEIS': 'DEISMultistepScheduler',
        'UniPC': 'UniPCMultistepScheduler',
        'PNDM': 'PNDMScheduler'
    }

    size_dict = {
            "512x512": (512, 512),
            "768x768": (768, 768)
        }


    class GenRequest(BaseModel):
        prompt: str = ("logo of The Witcher Portrait for a spiritual Premium Instagram    channel,artwork, fujicolor, "
                       "trending on artstation, faded forest background")
        negative_prompt: str = "bad, low res, ugly, deformed"
        width: int = 768
        height: int = 768
        num_inference_steps: int = 60
        guidance_scale: float = 5.0
        num_images_per_prompt: int = 1
        generator: int = 88300
        pipeline: str = pipeline_map[model_map["Stable Diffusion XL"]]
        scheduler: str = scheduler_map["PNDM"]
        modelid: str = model_map["Stable Diffusion XL"]


    default_values = GenRequest()
    ```
    
    For simplicity, we have defined these parameters directly in the client manually. However, keep in mind that, in reality, we could define them from the backend. Endpoints could be created in the API that the client can query to dynamically populate these parameters in a more decoupled manner. This information could also be defined in-memory or stored in databases.
    
* **utils.py**. It contains functions that perform secondary tasks, such as making requests to the server.

    ```python
    import httpx

    from config import size_dict


    async def generate_images(api_endpoint, params):
        async with httpx.AsyncClient() as client:
            response = await client.post(api_endpoint, json=params, timeout=120)
        return response


    def get_dimensions(size):
        if size in size_dict:
            return size_dict[size]
        else:
            raise ValueError(f"Invalid size: {size}")
    ```
    
* **client.py**. It contains the implementation of the interface with Streamlit.

```python
import asyncio

import numpy as np
import streamlit as st

from config import GenRequest, default_values, model_map, pipeline_map, scheduler_map
from utils import generate_images, get_dimensions

# Check if the session state object exists
if "generated_images" not in st.session_state:
    st.session_state.generated_images = []

if "generation_in_progress" not in st.session_state:
    st.session_state.generation_in_progress = False

if "seed_value" not in st.session_state:
    st.session_state.seed_value = default_values.generator


# Function to store image in the session state
def store_image(image, prompt):
    st.session_state.generated_images.append({
        "bytes": image,
        "prompt": prompt
    })
    st.session_state.generated_images = st.session_state.generated_images[-5:]


# Function to change the generation button status: enabled/disabled
def swap_generation_button_status():
    st.session_state.generation_in_progress = not st.session_state.generation_in_progress


# Function to generate a random seed
def generate_seed():
    st.session_state.seed_value = int(np.random.randint(1, 10 ** 10))


# Streamlit layout
st.title("Image Generator with Diffusion Models")

# -------------------------------------------------------------------------------------------------------
# SIDE BAR
# -------------------------------------------------------------------------------------------------------

st.sidebar.header("Model")

# Model
model_list = list(model_map.keys())
model_selected = st.sidebar.selectbox("Model", model_list,
                                      index=model_list.index("Stable Diffusion XL"))
model = model_map[model_selected]
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Scheduler
schedulers_list = list(scheduler_map.keys())
scheduler_option = st.sidebar.selectbox("Scheduler", schedulers_list,
                                        index=schedulers_list.index("PNDM"))
scheduler = scheduler_map[scheduler_option]
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Pipeline
pipeline = pipeline_map[model]

st.sidebar.header("Parameters")

# Prompt
prompt = st.sidebar.text_area("Prompt", default_values.prompt)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Negative Prompt
negative_prompt = st.sidebar.text_area("Negative Prompt", default_values.negative_prompt)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Number of Images
num_images_per_prompt = 1

# Image Size
size = st.sidebar.radio("Size", ("512x512", "768x768"), index=1)
witdh, height = get_dimensions(size)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Number of Inference Steps
num_inference_steps = st.sidebar.slider("Number of Inference Steps", min_value=1, max_value=200,
                                        value=default_values.num_inference_steps, step=1)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Guidance Scale
guidance_scale = st.sidebar.slider("Guidance Scale", min_value=0., max_value=20., value=default_values.guidance_scale,
                                   step=0.5, format="%.1f")
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# ********** Seed Options **********#
st.sidebar.markdown("#### Seed Options")

# Placeholder for dynamic UI update
seed_placeholder = st.sidebar.empty()

# Checkbox for Generate Random Seed
generate_random_seed = st.sidebar.checkbox("Generate Random Seed")

if generate_random_seed:
    # Generate random seed and update the input field
    seed_value = st.session_state.seed_value
    seed_value_input = seed_placeholder.number_input("Seed", value=seed_value, step=1)
    # Button to reload seed
    reload_button = st.sidebar.button("Reload Seed", key="reload_button", on_click=generate_seed)
else:
    # User input for seed value
    seed_value = seed_placeholder.number_input("Seed", min_value=1, max_value=10 ** 10,
                                               value=st.session_state.seed_value)
    seed_value_input = None

# Add a blank line in the sidebar
st.sidebar.markdown("<br><br>", unsafe_allow_html=True)

# Button to trigger image generation
generation_button = st.sidebar.button("Generate Images", type="primary",
                                      on_click=swap_generation_button_status,
                                      disabled=st.session_state.generation_in_progress)

# -------------------------------------------------------------------------------------------------------
# MAIN VIEW
# -------------------------------------------------------------------------------------------------------

# Image placeholder
# Check if there are existing generated images
if st.session_state.generated_images:
    # Show the last generated image
    last_image = st.session_state.generated_images[-1]
    placeholder = st.image(last_image["bytes"], caption=prompt, use_column_width=True)
else:
    # If no existing images, show the placeholder
    placeholder = st.image("https://www.pulsecarshalton.co.uk/wp-content/uploads/2016/08/jk-placeholder-image.jpg")

# Button to trigger image generation
if generation_button:

    # Set generation in progress to True
    st.session_state.generation_in_progress = True

    # API endpoint to generate images
    api_endpoint = "http://server:8000/generate"

    # Parameters to be sent to the API
    params = GenRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=witdh,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        generator=seed_value,
        pipeline=pipeline,
        scheduler=scheduler,
        modelid=model
    )

    # Clear placeholder
    placeholder.empty()

    with st.container():
        with st.spinner("Generating image..."):
            # Call the asynchronous function and wait for it to complete
            response = asyncio.run(generate_images(api_endpoint, params.model_dump()))

        if response.status_code == 200:
            # Display the generated image
            image = response.content
            st.image(image, caption=prompt, use_column_width=True)
            store_image(image, prompt)
            # Reset generation in progress
            swap_generation_button_status()
            # Generate a new seed if random seed is enabled
            if generate_random_seed:
                generate_seed()
        else:
            st.error(f"Error: {response.status_code}. Failed to generate images.")
            # Reset generation in progress
            swap_generation_button_status()

    st.rerun()

with st.container():
    st.markdown("----", unsafe_allow_html=True)
    st.markdown("### Previous results")
    for i, img in enumerate(st.session_state.generated_images[::-1][1:]):
        # Display image & prompt
        st.image(img["bytes"])
        st.markdown(img["prompt"])
```

Basically, the interface consists of a sidebar where the parameters controlling the generated image are located, a central area that is divided into two parts: one displaying the result of the generation and another showing the history of generations (the last 5). This history is saved in memory, in the client's session, meaning it is a temporary history. If you wanted to save it permanently, other alternatives could be considered, such as an S3 bucket in AWS or a Minio server.

### Running the project for the first time

To run the project, you have to do the following:
* To Have Docker installed on your machine. If you have Linux, you can follow the installation instructions from the official Docker website. Alternatively, you can install Docker Desktop on Linux or Mac. In the references, you will find links that will assist you.
* Clone the repository: `git clone https://github.com/curibe/ml-production-celery.git`
* Switch to the `front-back-no-celery` branch, which contains the current version of the code.
* Compile the project: `docker compose build`
* Run the project: `docker compose up`

When you run the project, both services are launched: the client and the server. By default, all parameters in the sidebar will be set to their default values. This means you just can click on the generate button and obtain an image for those parameters. This is how the interface made in Streamlit would look.

![image](https://hackmd.io/_uploads/r160XQovC.png)

Two images are shown to illustrate the functionality of displaying the history of generations. The first generated image (bottom one) was made with the default parameters, while the second (top one) was made by changing the sampler or scheduler to Euler ancestral.

Also, if you want to interact with your API directly, you can open the FastAPI docs (https://localhost:8000/docs) and run a test on the ‘generate’ endpoint through this interface. Remember that default values were defined for the request, so you just have to execute and wait for the image to be generated.

![image](https://hackmd.io/_uploads/ryzNNmiv0.png)

If you have run the project, you will have noticed two things: image generation with the default diffusion model uses approximately 7GB of VRAM, and it takes a certain amount of time to generate the image. Based on this, we could ask ourselves:
* What would happen if multiple users use the interface and send requests?
* If you want to process more than one request at a time to better utilize your GPU resources or perhaps control the number of concurrent image generations that can be processed due to limited VRAM.

Our  API would operate as follows: If user1 sends a request and then user2 sends another request, the API queues the user2 request while processing the user1 request. This queuing happens because the default Uvicorn configuration runs in a single-process, single-thread mode. Consequently, even with async endpoints, requests are queued. Additionally, since the process runs in a single thread in our case, both users will receive responses simultaneously, regardless of the order in which the requests were made. This means that user1 has to wait for approximately the time it takes to process two requests to receive a response. The reason behind this is our use of StreamingResponse, which does not send any data until the view function has finished executing. In essence, StreamingResponse buffers all the data until the view function completes, and it does not stream anything while the function is still executing.


To solve that we can implement these strategies:

* **Run additional workers for uvicorn**. This enables the processing of more concurrent requests. Since each worker operates in a separate thread, the StreamingResponse returns the image as soon as the view function receives the result from the ML  pipeline. However, running two workers simultaneously,  for instance, implies that GPU will be used simultaneously for both workers. This results in a cumulative GPU memory usage of 14GB. If your system lacks sufficient resources, it would be a problem because your GPU might run out of memory.
* **Run the ML pipeline as a background task**. This is a widely recommended strategy for tasks that involve significant processing time. The idea is to relieve the API of the processing burden and delegate it to a background task queue that handles the tasks asynchronously and concurrently. This approach allows the API to provide an initial response quickly and update it later using a websocket or long poll mechanism. In this case, we can implement a solution with [dramatiq](https://dramatiq.io/) or [Celery](https://docs.celeryq.dev/en/stable/) for handling heavy background computation tasks that do not require running in the same process as the API or indeed in the same server

In our case, we will implement a solution using Celery to run the pipeline as an independent task separate from the API. In the next section, we will explore how to carry out this implementation.


## Integrating Celery for Efficient Task Management: using standard tasks

<small>The code implemented for this section can be found at: https://github.com/curibe/ml-production-celery/tree/integrate-celery-default-task</small>

Celery is a distributed task queue system that allows us to offload time-consuming tasks to separate workers. By using Celery, we can distribute the workload across multiple machines, improving the overall performance and responsiveness of our ML application.
At this point, our initial code is working, it is capable of generating an Image using diffusion models tunning its params by a Streamlit interface. Now we want to integrate Celery to implement a background task system that allows us to separate the image generation pipeline from the API, and demonstrate its task management capabilities
We are going to handle the integration of Celery as commonly done: in the same base code of the server. This will allow us to view Celery as an extension of our app and thus enable a faster integration. However, keep in mind that Celery workers, responsible for performing heavy tasks, will be components that run separately from the server. In other words, we are going to approach this part as an event system, where our API will act as a Producer, while the workers will act as Consumers, communicating with each other through a broker responsible for queuing and distributing tasks.

![image](https://hackmd.io/_uploads/ryZySmoDC.png)

Taking into account that tasks in Celery are handled as functions, we will proceed to make the following modifications to the code from the previous section.

On the server:

1. We are going to create a folder inside `server/app` called ***celery*** that will contain the Celery integration: the definition of the Celery app with its configuration and the definition of tasks. These tasks are the ones that will now execute the ML pipeline.
2. We are going to create a new integration for Celery in `server/app/integrations`. Instead of calling the operator, which is responsible for executing the ML pipeline, it will send the task to Celery so that the workers can perform the task. At this point, the API delegates that task, freeing itself for another request.
3. We are going to create a new endpoint in the API for requests that will be processed with Celery. The idea is to have an app that allows us to choose whether to use a distributed task system or not. This endpoint no longer returns an image but a task ID.
4. Let's create an endpoint in the API to check the status of the task and obtain the final result.

On the client:

* We are going to create a dictionary for the API URL that allows us to choose whether to use distributed tasks or not.
* We are going to create a function that allows us to perform long polling in order to continuously check the status of the task and ultimately obtain the result.
Adapt the client to make this additional request.

For the deployment
* We are going to create three services in Docker Compose: one to run the Celery workers, another to run Redis, which will be the broker for our application for simplicity. However, we can choose other brokers like RabbitMQ for this purpose. And finally, one to monitor the Celery tasks through a UI. 

Let's examine each of these modifications in more detail.

### Define the Celery application and tasks.
We are going to create the following files in server/app/celery: `celery_app.py` and `task.py`.


**celery_app.py:**
```python
from celery import Celery
from kombu import Exchange, Queue

from app.config import get_settings

settings = get_settings()

# Define queue names
stable_diffusion_queue_name = settings.celery_task_queue
stable_diffusion_exchange_name = settings.celery_task_queue
stable_diffusion_routing_key = settings.celery_task_queue
worker_prefetch_multiplier = settings.celery_worker_prefetch_multiplier

# get all task modules
task_modules = ["app.celery.tasks"]

app = Celery(__name__)
app.conf.broker_url = settings.celery_broker_url
app.conf.result_backend = settings.celery_backend_url

# define exchanges
stable_diffusion_exchange = Exchange(stable_diffusion_exchange_name, type="direct")

# define queues
stable_diffusion_queue = Queue(
    stable_diffusion_queue_name,
    stable_diffusion_exchange,
    routing_key=stable_diffusion_routing_key,
)

# set the task queues
app.conf.task_queues = (
    stable_diffusion_queue,
)

# set the task routes
app.conf.task_routes = {
    "app.celery.tasks.*": {"queue": stable_diffusion_queue_name},
}

# serializer and accept content
app.conf.task_serializer = "pickle"
app.conf.result_serializer = "pickle"
app.conf.accept_content = ["application/json", "application/x-python-serialize"]


app.autodiscover_tasks(task_modules)

app.conf.worker_prefetch_multiplier = worker_prefetch_multiplier
```

Commonly, it is sufficient to write `app = Celery(name, broker, backend, include=[celery tasks])` to define the Celery application. However, I wanted to show you a longer but more detailed way that allows you to have more control over the configuration. In this file, we define several parameters:

* **Broker**. This specifies the message broker used by Celery to send and receive tasks. Common choices are Redis and RabbitMQ.
* **Backend**. This configures the result backend used by Celery to store task results. Also often Redis or database.
* **Exchange**. Represents an entity in the messaging system that routes messages to different queues based on specific criteria.
* **Queue**. Represents a storage location for tasks waiting to be executed.
* **Routing Key**. A label attached to a message that determines its destination queue. It will ensure Stable Diffusion tasks published to the exchange are routed to the Stable Diffusion queue.
* **Task queue and routes**. Defines the list of available queues. In this case, only the stable_diffusion_queue is defined. The routes specify which queue each task should be sent to. All tasks from modules in task_modules are routed to the stable_diffusion_queue.
* **Serializers**. It sets the task and result serializers to "pickle" and accepts content in both JSON and Python serialization formats.
* **Autodiscover tasks**. This command instructs Celery to automatically discover tasks in the specified modules. This is a convenient way to register tasks without explicitly listing them.
* **Worker Prefetch Multiplier**. Controls how many tasks a worker requests at once from broker, multiplied by the number of concurrent workers.

**task.py:**
```python
from app.celery.celery_app import app
from app.models.schemas import GenRequest
from app.operators.stable_diffusion import StableDiffusionText2Image


@app.task
def task_generate_image_with_text2img(*, request: GenRequest) -> str:
    generator = StableDiffusionText2Image(model_name=request.modelid, scheduler=request.scheduler,
                                          pipeline_name=request.pipeline)
    generation = generator.generate_images(request=request)
    return generation
```

We can see that essentially what was done was to move the execution of the pipeline from `GenerativeAIGenerator`, and it was set up as a task. Now, the task will be responsible for carrying out the image generation process.

### Define the Celery integration.

We are going to define a new integration for Celery within `server/app/integrations/genai_generator.py`.

**genai_generator.py:**
```python
from app.integrations.generation_interface import GenerationInterface
from app.models.schemas import GenRequest
from app.operators.stable_diffusion import StableDiffusionText2Image
from loguru import logger
from app.celery.tasks import task_generate_image_with_text2img


class GenerativeAIGenerator(GenerationInterface):
    …

class GenerativeAIGeneratorCelery(GenerationInterface):
    @staticmethod
    def generate_image_with_text2img(*, request: GenRequest) -> str:
        logger.info("generating image in Text2Image pipeline")
        taskid = task_generate_image_with_text2img.delay(request=request)
        return str(taskid)


```

We can see that the new generator, in this case, calls a Celery task and enqueues the request for Celery workers to handle. Celery automatically returns the task ID associated with the task. With this task ID, you can monitor the status of the task.

### Define the new endpoints in the API.

We are going to define two endpoints: one to receive requests for scheduling tasks in Celery and another to check the status of the task.

**app.py:**
```python
from io import BytesIO
from pathlib import Path

from celery.result import AsyncResult
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import get_settings
from app.config.logger import InitLogger
from app.integrations.genai_generator import GenerativeAIGenerator, GenerativeAIGeneratorCelery
from app.models.schemas import GenRequest
from app.services.generation_service import GenerationService
from app.utils.images import from_image_to_bytes

app = FastAPI()

# Create service to generate an image with Diffusion models
# We inject the image generator integration dependency in the service
generator_service = GenerationService(generator=GenerativeAIGenerator())
generator_service_celery = GenerationService(generator=GenerativeAIGeneratorCelery())

# intermediate code without changes
...



@app.post('/generate')
async def generate(request: GenRequest):
    ...


@app.post('/generate_async')
async def generate_async(request: GenRequest):
    # Call the service to generate the images according to the request params
    taskid = generator_service_celery.generate_images_with_text2img(request=request)
    return {"taskid": taskid}


@app.get(
    "/results/{task_id}",
)
async def get_generation_result(task_id, ):
    task = AsyncResult(task_id)

    if not task.ready():
        return JSONResponse(content={"task": str(task.status)}, status_code=202)

    result = task.get()
    img_bytes = from_image_to_bytes(result[0])
    return StreamingResponse(BytesIO(img_bytes), media_type="image/png")
```

Note that first, we need to create another instance of the `generator_service_celery` service, to which the new Celery generator has been injected as a dependency.

The `/generate_async` endpoint receives an image generation request from the client and responds with the task ID. The client will use this task ID to check the status of the task by sending a request with the task ID to the `/results/{task_id}` endpoint and obtain the image once the worker has finished the task.

In the `get_generation_result` function, we return two response formats: a JSON with the task status in case it has not finished, and the image as a StreamingResponse when the task is completed. To easily distinguish the response format on the client side, we make use of our friends, the HTTP Codes: if the task is still pending, we send a 202 as a response, and if the task has already finished, the StreamingResponse automatically sends a status 200.

It's worth highlighting that due to the architecture used for the project, it was easy to add a new functionality without the need to delete or modify existing ones.


### Implementation of changes in the client.

We are going to create the URL map that allows us to dynamically select in the client configuration whether to use the distributed task system or not:

**config.py:**
```python
from pydantic import BaseModel

TASK_TYPE = "CELERY"

url_map = {
    "CELERY": "http://server:8000/generate_async",
    "ASYNC": "http://server:8000/generate",
}

api_url = url_map.get(TASK_TYPE, "http://server:8000/generate")
result_url = "http://server:8000/results"

# The rest of the settings
.
.
.
```

With the `TASK_TYPE` variable, you can choose how you want to run your client: whether you want your image to be generated using Celery or the server.

It is also necessary to create the function that will perform long polling on the results endpoint to monitor the status of the task and obtain the image at the end.


**utils.py**
```python
import asyncio

import httpx

from config import size_dict


async def generate_images(api_endpoint, params):
    async with httpx.AsyncClient() as client:
        response = await client.post(api_endpoint, json=params, timeout=120)
    return response


async def long_poll_task_result(task_id, result_url, max_retries=20, delay=5):
    async with httpx.AsyncClient() as client:
        for _ in range(max_retries):
            response = await client.get(f"{result_url}/{task_id}")

            if response.status_code == 200:
                # The task is finished
                return response.content
            elif response.status_code == 202:
                # The task is not finished yet
                print(response.json())
            else:
                # Error occurred
                print(f"Error: {response.status_code}")
                return None

            # Wait before next retry
            await asyncio.sleep(delay)

        # Reached maximum retries
        return None


def get_dimensions(size):
    …

```

To conclude, let's integrate the long poll in the client to obtain the result of the task. We will continue using the `generate_images` function since it is the one sending the request and obtaining the task ID as a response:


**client.py**
```python
import asyncio

import numpy as np
import streamlit as st

from config import GenRequest, TASK_TYPE, api_url, default_values, model_map, pipeline_map, result_url, scheduler_map
from utils import generate_images, get_dimensions, long_poll_task_result


.
.
.

# Button to trigger image generation
if generation_button:

    # Set generation in progress to True
    st.session_state.generation_in_progress = True

    # Parameters to be sent to the API
    params = GenRequest(prompt=prompt, negative_prompt=negative_prompt, width=witdh, height=height,
                        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                        num_images_per_prompt=num_images_per_prompt, generator=seed_value, pipeline=pipeline,
                        scheduler=scheduler,
                        modelid=model)

    # Clear placeholder
    placeholder.empty()

    with st.container():
        with st.spinner("Generating image..."):
            # Call the asynchronous function and wait for it to complete
            response = asyncio.run(generate_images(api_url, params.model_dump()))

            if response.status_code == 200:

                if TASK_TYPE in ["CELERY"]:
                    # get the task id
                    task_id = response.json()["taskid"]

                    # Call the asynchronous function and wait for it to complete
                    image = asyncio.run(long_poll_task_result(task_id, result_url))
                else:
                    image = response.content

                # Display the generated image
                st.image(image, caption=prompt, use_column_width=True)
                store_image(image, prompt)
                # Reset generation in progress
                swap_generation_button_status()
                # Generate a new seed if random seed is enabled
                if generate_random_seed:
                    generate_seed()
            else:
                st.error(f"Error: {response.status_code}. Failed to generate images.")
                # Reset generation in progress
                swap_generation_button_status()

    st.rerun()
.
.
.

```

We have used the `TASK_TYPE` variable to evaluate whether Celery will be used. If Celery is used, the response contains the task ID, and thus, a long poll is performed. Otherwise, if Celery is not used, the response contains the image.

### Adjusting the deployment.

Finally, to be able to run our project with Celery, it is necessary to add the necessary services to the Docker Compose:

```yaml
version: "3.8"
services:
  server:
    build:
      context: server
      dockerfile: ./Dockerfile
    image: server:latest
    command: python3 main.py
    volumes:
      - ./server:/opt/api
    ports:
      - "8000:8000"
    stdin_open: true
    tty: true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [ gpu ]

  client:
    build:
      context: client
      dockerfile: ./Dockerfile
    image: streamlit-client:latest
    volumes:
      - ./client:/opt/app
    ports:
      - "8501:8501"
    stdin_open: true
    tty: true


  worker:
    build:
      context: server
      dockerfile: ./Dockerfile
    command: celery -A app.celery.celery_app worker --max-tasks-per-child=1 --loglevel=info -Q stable_diffusion
    volumes:
      - ./server:/opt/api
    stdin_open: true
    tty: true
    depends_on:
      server:
        condition: service_started
      redis:
        condition: service_started
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [ gpu ]

  flower:
    build: ./server
    command: celery flower --basic_auth=admin:password
    volumes:
      - ./server:/opt/api
    ports:
      - "5555:5555"
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
    depends_on:
      server:
        condition: service_started
      redis:
        condition: service_started

  redis:
    image: redis:6-alpine
    healthcheck:
      test: [ "CMD", "redis-cli", "ping" ]
      interval: 1s
      timeout: 3s
      retries: 30
```

We have added the worker service, which runs the Celery worker. You can include more options in the Celery command if needed. We also added the [flower](https://flower.readthedocs.io/en/latest/) service, which provides a graphical interface to monitor Celery processes.

==add flower screenshot==

In this way, your application is ready for you to generate your images using a distributed task system. This allows you to scale your application according to the available resources.

==check if add a video or gif==


## Conclusion

## References
