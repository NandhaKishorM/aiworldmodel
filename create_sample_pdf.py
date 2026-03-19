"""
Generates a multi-page PDF document for parametric injection testing.
Contains explicit, completely falsified 'facts' so we can verify if the model successfully 
memorizes them directly from the weights.
"""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

def generate_pdf(filename="sample_mission_briefing.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    # --- PAGE 1: The Target ---
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, "CONFIDENTIAL MISSION BRIEFING: OPERATION NIGHTFALL")
    
    c.setFont("Helvetica", 12)
    c.drawString(72, height - 120, "Page 1: The Primary Target")
    c.drawString(72, height - 144, "The primary target of Operation Nightfall is the submarine known as the 'Abyssal Leviathan'.")
    c.drawString(72, height - 168, "It is currently docked at the clandestine facility located in Sector 7-G.")
    c.drawString(72, height - 192, "Commander for this operation is General Atticus Vance.")
    c.showPage()

    # --- PAGE 2: Security Codes ---
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, "SECURITY & ACCESS PROTOCOLS")
    
    c.setFont("Helvetica", 12)
    c.drawString(72, height - 120, "Page 2: Submarine Access")
    c.drawString(72, height - 144, "To bypass the outer blast doors of the Abyssal Leviathan, you must use the sequence:")
    c.drawString(72, height - 168, "OVERRIDE-ECLIPSE-9932")
    c.drawString(72, height - 192, "Once inside, the communications relay wavelength is 445.8 MHz.")
    c.showPage()
    
    # --- PAGE 3: Evacuation ---
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, "EVACUATION STRATEGY")
    
    c.setFont("Helvetica", 12)
    c.drawString(72, height - 120, "Page 3: Extraction Point")
    c.drawString(72, height - 144, "Upon completion of the sabotage, operatives must head to the extraction zone.")
    c.drawString(72, height - 168, "The extraction zone is an abandoned lighthouse three miles north of the dock.")
    c.drawString(72, height - 192, "The password for the extraction pilot is 'Midnight Horizon'.")
    c.showPage()

    c.save()
    print(f"Generated multi-page PDF: {filename}")

if __name__ == "__main__":
    generate_pdf()
