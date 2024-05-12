# Scene Understanding

This project is a deep learning-based application for understanding and describing scenes in images. It uses a custom dataset and a model trained on this dataset to generate captions for images. The application includes a GUI for selecting images and generating captions.

The model composes of a MobileVIT feature extractor and a custom transfomer model for generating captions.

The dataset used is the COCO dataset, which contains images and corresponding captions. The model is trained on this dataset to generate captions for new images.

## Project Structure
The project is structured as follows:

* **code**: Contains the main Python scripts and Jupyter notebooks for the project.
* **models**: Contains the trained model files.
* **examples**: Contains example images for testing.

## Key Files
* code/dataset.py: Defines the CustomDataset class for loading and processing the image dataset.
* code/tokenizer.py: Contains functions for encoding and decoding text.
* code/model.py: Defines the Decoder class, which is the main model used for generating captions.
* code/training.py: Contains the training loop for the model.
* code/main.py: Contains the Loader class for running the model and the main GUI code for the application.

## Key Classes and Functions
* CustomDataset (code/dataset.py): A class for loading and processing the image dataset.
* Tokenizer (code/tokenizer.py): A class for encoding and decoding text.
* Decoder (code/model.py): The main model used for generating captions.
* Loader (code/main.py): A class for running the model and interacting with the GUI.

## How to Run
* Ensure you have the necessary dependencies installed. These include torch, torchvision, pandas, PIL, tkinter along with sv_ttk.
* Run the main.py script to start the application.
* Use the GUI to select an image and generate a caption.

## Model Training
The model can be trained by running the training.py script. The trained model will be saved to the models directory.

## Note
This project uses a custom tokenizer, which is loaded from tokenizer2.pkl. The tokenizer is used to convert the captions into a format that can be processed by the model.

## License 

This project is licensed under the GPL-V3 License - see the LICENSE.md file for details.