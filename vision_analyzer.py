import json
import os
from google.cloud import vision
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

class VisionAnalyzer:
    def __init__(self):
        # Get the absolute path to the credentials file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        credentials_path = os.path.join(current_dir, 'credentials.json')
        
        # Verify the file exists
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(f"Credentials file not found at: {credentials_path}")
            
        # Initialize the client with explicit credentials
        self.vision_client = vision.ImageAnnotatorClient.from_service_account_json(credentials_path)

    def analyze_thumbnail(self, image_url):
        """
        Analyzes a YouTube thumbnail image using Google Vision API.
        """
        try:
            # Download image from URL
            image_data = urlopen(image_url).read()
        except HTTPError as e:
            print(f"HTTPError: Unable to open URL {image_url}. Error code: {e.code}")
            return None
        except URLError as e:
            print(f"URLError: Failed to reach server for URL {image_url}. Reason: {e.reason}")
            return None
        except Exception as e:
            print(f"Unexpected error occurred while opening URL {image_url}: {e}")
            return None

        image = vision.Image(content=image_data)

        # Perform image analysis
        response = self.vision_client.annotate_image({
            'image': image,
            'features': [
                {"type": vision.Feature.Type.LABEL_DETECTION},  # General objects
                {"type": vision.Feature.Type.TEXT_DETECTION},   # OCR
                {"type": vision.Feature.Type.FACE_DETECTION},   # Face analysis
                {"type": vision.Feature.Type.IMAGE_PROPERTIES}  # Color analysis
            ]
        })

        # Extract information
        labels = [label.description for label in response.label_annotations]
        text = response.text_annotations[0].description if response.text_annotations else "No text detected"
        faces = len(response.face_annotations)  # Count faces
        colors = response.image_properties_annotation.dominant_colors.colors

        dominant_colors = [{"red": c.color.red, "green": c.color.green, "blue": c.color.blue} for c in colors[:3]]

        return {
            "labels": labels,
            "text": text,
            "num_faces": faces,
            "dominant_colors": dominant_colors
        }

    def analyze_videos(self, videos):
        analyzed_data = []
        for video in videos:
            analysis = self.analyze_thumbnail(video["snippet"]["thumbnails"]["high"]["url"])
            if analysis is not None:
                video["analysis"] = analysis
                analyzed_data.append(video)
        return analyzed_data

    def save_analysis_to_json(self, analyzed_data, filename="video_analysis.json"):
        with open(filename, 'w') as f:
            json.dump(analyzed_data, f, indent=4)

def load_video_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data["items"]

if __name__ == "__main__":
    # Load video data from JSON file
    video_data = load_video_data('youtube_data_ai_key.json')

    # Create an instance of VisionAnalyzer
    analyzer = VisionAnalyzer()

    # Analyze the videos
    analyzed_videos = analyzer.analyze_videos(video_data)

    # Save the analysis results to a JSON file
    analyzer.save_analysis_to_json(analyzed_videos, filename="analyzed_video_data.json")
