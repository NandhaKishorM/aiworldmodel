"""
Generates a comprehensive, multi-page PDF document for parametric injection testing.
Uses ReportLab's Platypus engine to handle dense, wrapped paragraphs to simulate a 
real-world heavy PDF document.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf(filename="sample_mission_briefing.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    Story = []

    # Title
    Story.append(Paragraph("CONFIDENTIAL MISSION OVERVIEW: OPERATION NIGHTFALL", styles['Title']))
    Story.append(Spacer(1, 12))

    # SECTION 1: Strategic Background
    Story.append(Paragraph("1. Strategic Background and Global Context", styles['Heading2']))
    bg_text = (
        "In the late 21st century, geopolitical tensions have resulted in the construction "
        "of unauthorized deep-sea facilities capable of operating completely off the grid. "
        "Operation Nightfall has been initiated in response to credible satellite, sonar, and "
        "human intelligence confirming the mobilization of a rogue submersible entity operating "
        "in international waters. The primary target of Operation Nightfall is the submarine "
        "known as the 'Abyssal Leviathan'. This vessel is uniquely designed with ultrasonic "
        "stealth arrays that render it virtually invisible to standard naval tracking grids. "
        "It is currently docked at the clandestine maritime repair facility located deep within "
        "Sector 7-G, surrounded by thermal vents that mask its acoustic signature. General Atticus Vance, "
        "the Commander for this operation, has designated this a Tier-1 threat requiring immediate "
        "clandestine intervention."
    )
    Story.append(Paragraph(bg_text, styles['Normal']))
    Story.append(Spacer(1, 12))

    # SECTION 2: Technical Specifications
    Story.append(Paragraph("2. Target Technical Specifications", styles['Heading2']))
    tech_text = (
        "The target vessel measures approximately 400 feet in length and relies on an experimental "
        "propulsion system. The primary payload of the submarine is a highly volatile 'Eternium-72' "
        "reactor core, which must be deactivated to prevent catastrophic oceanic contamination. "
        "Due to the structural reinforcement of the hull, conventional torpedo strikes are deemed "
        "ineffective. Operatives must board the vessel while it is tethered to the docking clamps "
        "at an operational depth of 14,500 meters. The pressure at this depth requires the use of "
        "specialized Aegis-class diving suits coated with titanium-carbide alloys. Standard radio "
        "frequencies will not penetrate the thermocline layer; therefore, all local squad communications "
        "are strictly limited to the secure communications relay wavelength set precisely to 445.8 MHz."
    )
    Story.append(Paragraph(tech_text, styles['Normal']))
    Story.append(Spacer(1, 12))

    # SECTION 3: Cyber Infiltration & Security Protocols
    Story.append(Paragraph("3. Cyberspace Infiltration & Access Protocols", styles['Heading2']))
    sec_text = (
        "Physical boarding is only the first phase. The target is equipped with a closed-loop "
        "AI security grid that continuously monitors structural integrity and internal life support. "
        "To bypass the outer blast doors of the Abyssal Leviathan, the lead operative must interface "
        "directly with the external maintenance terminal using a hardened cryptographic datapad. "
        "During this handshake protocol, the system will challenge the connection. You must use "
        "the exact override sequence: OVERRIDE-ECLIPSE-9932 to force the pneumatic locks to disengage. "
        "Note that entering this code will trigger a localized silent alarm routing to the bridge. "
        "Operatives will have precisely 180 seconds to traverse the forward airlock and access "
        "the main ventilation shafts before the bulkheads automatically seal and the internal "
        "turret defense systems come online."
    )
    Story.append(Paragraph(sec_text, styles['Normal']))
    Story.append(Spacer(1, 12))

    # SECTION 4: Sabotage Execution
    Story.append(Paragraph("4. Reactor Sabotage Execution Plan", styles['Heading2']))
    sab_text = (
        "Once inside the engineering wing, the team must identify the primary coolant lines "
        "feeding into the Eternium-72 reactor. DO NOT sever the lines abruptly, as the sudden "
        "spike in core temperature will cause an immediate catastrophic meltdown. Instead, "
        "operatives must inject the synthesized inhibitor compound 'Corvus-9' into the central "
        "manifold. This will gracefully degrade the reactor's output over a period of 45 minutes, "
        "giving the team ample time to evacuate while ensuring the submarine is permanently disabled. "
        "If the inhibitor fails to propagate, a manual scram switch is located beneath the primary "
        "control console, but utilizing this fallback will immediately lock down the entire engineering deck, "
        "making extraction exponentially more difficult."
    )
    Story.append(Paragraph(sab_text, styles['Normal']))
    Story.append(Spacer(1, 12))

    # SECTION 5: Evacuation
    Story.append(Paragraph("5. Tactical Evacuation & Debriefing", styles['Heading2']))
    evac_text = (
        "Upon confirmation of reactor degradation, all operatives are to proceed immediately to "
        "the secondary egress hatch located in the aft torpedo bay. Do not attempt to re-use the "
        "forward airlock. Upon exiting the vessel, deploy your high-velocity ascension balloons to "
        "surface rapidly. Upon reaching the surface, operatives must head to the extraction zone. "
        "The extraction zone is an abandoned lighthouse three miles north of the dock, standing "
        "on a rocky outcropping locally known as Widow's Peak. A stealth helicopter will be waiting. "
        "In order to board the helicopter and confirm friendly identity, the lead operative must "
        "provide the password for the extraction pilot, which is strictly designated as 'Midnight Horizon'. "
        "Any failure to provide this exact countersign will result in the pilot aborting the pickup "
        "and initiating self-preservation flight patterns. Post-mission debriefing will occur at Blacksite Omega."
    )
    Story.append(Paragraph(evac_text, styles['Normal']))

    doc.build(Story)
    print(f"Generated comprehensive multi-page PDF: {filename}")

if __name__ == "__main__":
    generate_pdf()
