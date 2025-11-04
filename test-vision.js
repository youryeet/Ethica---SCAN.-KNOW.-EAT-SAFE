import vision from "@google-cloud/vision";
import fs from "fs";

const client = new vision.ImageAnnotatorClient();

async function testOCR() {
  try {
    // Load an image from your local disk
    const imagePath = "./test-image.jpg"; // <-- replace with a small test image path
    const imageBuffer = fs.readFileSync(imagePath);

    const [result] = await client.textDetection({ image: { content: imageBuffer } });
    const detections = result.textAnnotations;

    if (!detections || detections.length === 0) {
      console.log("No text detected!");
    } else {
      console.log("Text detected:\n", detections[0].description);
    }
  } catch (err) {
    console.error("Error with Google Vision API:", err);
  }
}

testOCR();

