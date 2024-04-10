FROM pytorch/pytorch:latest

WORKDIR /app

# Copy the model and prediction script
COPY ./checkpoints/cats_dogs_classifier_v1/1.pt .
COPY model.py .
COPY predict.py .
COPY preprocessing.py .
COPY ./images .
COPY cat.3880.jpg .
COPY dog.3880.jpg .

# Install additional dependencies if needed
RUN pip install gunicorn pillow

# Expose port for gunicorn
EXPOSE 8000

# Command to start the server
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "predict:app"]