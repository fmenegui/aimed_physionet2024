"""Create artificial drawings and placeholder labels for software checks only."""
import argparse
import math
from pathlib import Path
from PIL import Image, ImageDraw


def make_data(output, records=20):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if records < 1:
        raise ValueError("records must be positive")
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    for record in range(records):
        name = f"demo{record:03d}"
        image = Image.new("RGB", (768, 576), "white")
        draw = ImageDraw.Draw(image)
        for x in range(0, 768, 8):
            draw.line((x, 0, x, 576), fill=(245, 211, 211), width=1)
        for y in range(0, 576, 8):
            draw.line((0, y, 768, y), fill=(245, 211, 211), width=1)
        draw.rectangle((5, 5, 520, 25), fill="white")
        draw.text((8, 8), "ARTIFICIAL DEMO. NOT A PATIENT ECG. PLACEHOLDER LABELS.", fill="black")
        for lead, lead_name in enumerate(leads):
            row, col = divmod(lead, 4)
            x0, y0 = 12 + col * 189, 105 + row * 170
            draw.text((x0, y0 - 50), lead_name, fill="black")
            points = []
            for x in range(175):
                phase = ((x + record) % 58) / 58
                wave = (0.13 * math.exp(-((phase - 0.2) / 0.05) ** 2)
                        - 0.18 * math.exp(-((phase - 0.40) / 0.025) ** 2)
                        + math.exp(-((phase - 0.43) / 0.018) ** 2)
                        - 0.25 * math.exp(-((phase - 0.47) / 0.025) ** 2)
                        + 0.25 * math.exp(-((phase - 0.7) / 0.09) ** 2))
                points.append((x0 + x, y0 - (32 + lead) * wave))
            draw.line(points, fill=(25, 25, 25), width=2)
        image.save(output / f"{name}-0.png")
        # These arbitrary labels exercise multilabel software, not diagnosis.
        label = "NORM" if record % 2 == 0 else "AFIB/AFL"
        header = f"{name} 12 500 5000\n"
        header += "\n".join(f"{name}.dat 16 1000/mV 16 0 0 0 0 {lead}" for lead in leads)
        header += f"\n# Labels: {label}\n# Image: {name}-0.png\n"
        (output / f"{name}.hea").write_text(header)
    print(f"Created {records} artificial records in {output}. Software checks only.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/demo")
    parser.add_argument("--records", type=int, default=20)
    args = parser.parse_args()
    make_data(args.output, args.records)
