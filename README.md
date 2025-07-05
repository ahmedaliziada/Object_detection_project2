# Object Tracking App

A professional, interactive web application for object tracking in videos using background subtraction, built with **Streamlit** and **OpenCV**.

## Features

- 🎥 **Upload and process video files** (`mp4`, `avi`, `mov`)
- 🟦 **Real-time object detection** using background subtraction
- 🎨 **Customizable detection box color** (choose any color)
- ⏩ **Adjustable playback speed** (1–60 frames/sec)
- 🖼️ **Side-by-side display** of original and processed frames
- 📊 **Progress bar** for video processing feedback
- 💻 **Modern, responsive UI** with sidebar controls

## Demo

![App Screenshot](![alt text](image-1.png))
## Getting Started

### Prerequisites

- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/)

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/object-tracking-app.git
    cd object-tracking-app
    ```

2. **Install dependencies:**
    ```bash
    pip install streamlit opencv-python numpy
    ```

### Running the App

```bash
streamlit run app.py
```

Open your browser and go to [http://localhost:8501](http://localhost:8501).

## Usage

1. Upload a video file (`mp4`, `avi`, or `mov`).
2. Pick your preferred detection box color from the sidebar.
3. Adjust the playback speed as needed.
4. Watch the original and detected frames side by side.

## Project Structure

```
object-tracking-app/
├── app.py
├── README.md
└── requirements.txt
```

## License

This project is licensed under the ROUTE License.

---

<center>Made with ❤️ using Streamlit & OpenCV | 2025</center>