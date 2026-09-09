"""
Your portfolio case studies, structured for embedding into Pinecone.
Add/edit entries here as you complete new projects, then re-run rag_setup.py.
"""

CASE_STUDIES = [
    {
        "id": "rehbar-real-estate",
        "title": "Rehbar Real Estate - Voice Agent 'Alex'",
        "industry": "Real Estate",
        "summary": (
            "Built a real estate voice agent (Vapi, n8n, Google Calendar, Google Sheets, "
            "Deepgram) that handles inbound property inquiries, books viewings directly "
            "into the agent's calendar, and logs every lead automatically."
        ),
    },
    {
        "id": "apex-injury-law",
        "title": "Apex Injury Law - Legal Intake Agent",
        "industry": "Legal / Personal Injury",
        "summary": (
            "Built a legal intake voice agent that qualifies personal injury cases, "
            "generates intake documents via SignWell, sends Gmail alerts to the firm, "
            "and confirms appointments through a Netlify-hosted form."
        ),
    },
    {
        "id": "careloop-health",
        "title": "CareLoop Health - Clinic Dashboard + Voice Agent 'Ava'",
        "industry": "Healthcare",
        "summary": (
            "Built a 13-page clinic operations dashboard backed by 8 n8n workflows and a "
            "Vapi voice agent ('Ava') handling patient scheduling, reminders, and intake."
        ),
    },
    {
        "id": "meridian-group",
        "title": "Meridian Group - Enterprise AI Receptionist",
        "industry": "Enterprise / Multi-department",
        "summary": (
            "Built a 6-department AI receptionist (Vapi, n8n, Supabase, HubSpot) routing "
            "calls and inquiries to the correct department automatically with CRM logging."
        ),
    },
    {
        "id": "glowlab",
        "title": "GlowLab - Skincare E-commerce Chatbot 'Luna'",
        "industry": "E-commerce / Retail",
        "summary": (
            "Built an e-commerce chatbot (n8n, Groq, Netlify) that answers product "
            "questions, recommends skincare routines, and guides customers to checkout."
        ),
    },
    {
        "id": "whatsapp-ai-automation",
        "title": "WhatsApp AI Automation - Support System",
        "industry": "Customer Support / SMB",
        "summary": (
            "Built a WhatsApp support automation (Green API, n8n, Groq Whisper, Supabase, "
            "HubSpot) that transcribes voice notes, answers common questions, and escalates "
            "complex issues to a human via Slack."
        ),
    },
    {
        "id": "flavor-haven",
        "title": "Flavor Haven - Restaurant AI Receptionist 'Adam'",
        "industry": "Hospitality / Restaurants",
        "summary": (
            "Built a restaurant voice receptionist (Vapi, n8n, Google Sheets, Twilio SMS) "
            "that takes reservations, answers menu questions, and sends SMS confirmations."
        ),
    },
]
