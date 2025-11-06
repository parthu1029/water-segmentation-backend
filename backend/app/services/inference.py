import os
import numpy as np
try:
    import importlib
    tf = importlib.import_module("tensorflow")
except Exception:
    tf = None
try:
    import rasterio
    RASTERIO_AVAILABLE = True
except Exception:
    rasterio = None
    RASTERIO_AVAILABLE = False
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    Image = None
    PIL_AVAILABLE = False
import io

class ModelInference:
    """Handles model loading and inference for water segmentation"""
    
    def __init__(self, model_path):
        """
        Initialize the model inference service.
        
        Args:
            model_path: Path to the saved TensorFlow model
        """
        self.model_path = model_path
        self.model = None
        self.input_size = (256, 256)  # Expected input size for the model
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the TensorFlow model"""
        try:
            if tf is None:
                print("TensorFlow not available; skipping model load. Falling back to NDWI threshold.")
                self.model = None
                return
            if not os.path.exists(self.model_path):
                print(f"Warning: Model not found at {self.model_path}")
                return
                
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            print(f"Model loaded successfully from {self.model_path}")
            self.model.summary()
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None
    
    def preprocess_image(self, image_path):
        """
        Preprocess the input image for the model.
        
        Args:
            image_path: Path to the input image (GeoTIFF)
            
        Returns:
            Preprocessed image as a numpy array
        """
        try:
            if tf is None:
                raise RuntimeError("TensorFlow not available to preprocess image for model")
            if not RASTERIO_AVAILABLE:
                raise RuntimeError("rasterio not available to read GeoTIFF for model preprocessing")
            # Read the image using rasterio
            with rasterio.open(image_path) as src:
                # Read the first 3 bands (RGB)
                image = src.read([1, 2, 3])  # shape (C,H,W)

                # Convert to float32
                image = image.astype(np.float32)

                # If values look like 0..255, normalize to 0..1. If already 0..1, keep.
                max_val = float(image.max()) if image.size > 0 else 1.0
                if max_val > 1.5:
                    image = image / 255.0

                # Transpose to (H,W,C)
                image = np.transpose(image, (1, 2, 0))

                # Resize to model input size (requires TensorFlow)
                image = tf.image.resize(image, self.input_size)

                # Add batch dimension
                image = tf.expand_dims(image, axis=0)

                return image
                
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            raise
    
    def predict(self, image):
        """
        Run inference on the input image.
        
        Args:
            image: Preprocessed input image
            
        Returns:
            Segmentation mask as a numpy array
        """
        if self.model is None:
            raise ValueError("Model not loaded. Cannot perform inference.")
            
        try:
            # Run inference
            prediction = self.model.predict(image)
            
            # Remove batch dimension and get the first (and only) prediction
            mask = prediction[0]
            
            # Apply threshold to get binary mask (adjust threshold as needed)
            binary_mask = (mask > 0.5).astype(np.uint8)
            
            return binary_mask
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise
    
    @staticmethod
    def save_mask_as_png(mask, output_path):
        """
        Save the segmentation mask as a PNG file.
        
        Args:
            mask: Segmentation mask as a numpy array
            output_path: Path to save the output PNG
        """
        try:
            if not PIL_AVAILABLE or Image is None:
                raise RuntimeError("Pillow (PIL) is not available to save PNG files")
            # Convert to uint8 and scale to 0-255
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # Convert to PIL Image and save
            mask_image = Image.fromarray(mask_uint8.squeeze(), mode='L')
            mask_image.save(output_path, 'PNG')
            
            print(f"Mask saved to {output_path}")
            
        except Exception as e:
            print(f"Error saving mask: {str(e)}")
            raise

    @staticmethod
    def save_mask_as_overlay_png(mask, output_path, color=(0, 150, 255, 255)):
        """
        Save a binary mask as a transparent overlay PNG, coloring water pixels.

        Args:
            mask: Binary segmentation mask (H,W) or (H,W,1) with values {0,1}
            output_path: Output path for PNG
            color: RGBA tuple for water pixels
        """
        try:
            if not PIL_AVAILABLE or Image is None:
                raise RuntimeError("Pillow (PIL) is not available to save PNG files")
            m = mask.squeeze().astype(np.uint8)
            h, w = m.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[m > 0, 0] = color[0]
            rgba[m > 0, 1] = color[1]
            rgba[m > 0, 2] = color[2]
            rgba[m > 0, 3] = color[3]
            Image.fromarray(rgba, mode='RGBA').save(output_path, 'PNG')
            print(f"Overlay mask saved to {output_path}")
        except Exception as e:
            print(f"Error saving overlay mask: {str(e)}")
            raise
    
    def calculate_water_statistics(self, mask, pixel_size=10):
        """
        Calculate water statistics from the segmentation mask.
        
        Args:
            mask: Segmentation mask as a numpy array
            pixel_size: Size of each pixel in meters
            
        Returns:
            Dictionary containing water statistics
        """
        try:
            # Calculate pixel areas
            pixel_area = (pixel_size ** 2) / 1e6  # Convert to km²
            
            # Count water pixels
            water_pixels = np.sum(mask > 0)
            total_pixels = mask.size
            
            # Calculate areas
            water_area = water_pixels * pixel_area
            total_area = total_pixels * pixel_area
            water_percentage = (water_pixels / total_pixels) * 100 if total_pixels > 0 else 0
            
            return {
                'water_pixels': int(water_pixels),
                'total_pixels': int(total_pixels),
                'water_area_km2': float(water_area),
                'total_area_km2': float(total_area),
                'water_percentage': float(water_percentage)
            }
            
        except Exception as e:
            print(f"Error calculating water statistics: {str(e)}")
            return {
                'water_pixels': 0,
                'total_pixels': 0,
                'water_area_km2': 0.0,
                'total_area_km2': 0.0,
                'water_percentage': 0.0
            }
