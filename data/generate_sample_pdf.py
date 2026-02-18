"""Generate sample.pdf for testing. Run once: python data/generate_sample_pdf.py"""

from pathlib import Path

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
except ImportError:
    # Fallback: create a minimal PDF manually
    content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj
4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
5 0 obj<</Length 44>>stream
BT /F1 12 Tf 72 720 Td (See sample.txt for full content.) Tj ET
endstream
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000340 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
434
%%EOF"""
    Path(__file__).with_name("sample.pdf").write_bytes(content)
    print("Created minimal sample.pdf (install reportlab for full version)")
    exit(0)

OUTPUT = Path(__file__).with_name("sample.pdf")

REPORT_TEXT = """
ACME CORP — Q3 2025 QUARTERLY REPORT

Prepared by: Finance & Strategy Team
Date: October 15, 2025

EXECUTIVE SUMMARY

Acme Corp delivered strong results in Q3 2025, with annual recurring revenue (ARR) reaching $28 million, representing 40% year-over-year growth. The company added 45 new enterprise customers during the quarter, bringing total customer count to 312. Net revenue retention rate remained healthy at 118%.

FINANCIAL HIGHLIGHTS

Revenue: $7.2 million (Q3 2025), up from $5.1 million (Q3 2024). Gross margin improved to 78%, up from 74% in the prior year, driven by infrastructure optimization and improved unit economics.

Operating expenses totaled $6.8 million, with R&D accounting for 52% ($3.5M), Sales & Marketing at 30% ($2.0M), and G&A at 18% ($1.3M). The company achieved positive operating cash flow of $0.4 million for the first time.

Cash position: $18.5 million remaining from Series B funding, providing approximately 24 months of runway at current burn rate.

PRODUCT UPDATES

AcmeFlow (flagship product): Released version 3.0 with real-time collaboration features, custom workflow templates, and enhanced API. Customer adoption of v3.0 reached 68% within 6 weeks of launch.

AcmeGuard (security module): Launched in beta with 25 pilot customers. Early feedback indicates strong demand for automated compliance reporting. General availability planned for Q1 2026.

AcmeInsight (analytics dashboard): New product in development. Alpha testing with 10 design partners. Expected beta launch in Q4 2025.

CUSTOMER METRICS

Total customers: 312 (up from 267 in Q2 2025)
Enterprise customers (>$100K ARR): 28 (up from 22)
Average contract value: $89,700 (up 15% YoY)
Net revenue retention: 118%
Gross churn: 3.2% (annualized)
Customer satisfaction (NPS): 62

Top new customers in Q3: GlobalTech Industries, Meridian Healthcare, Pacific Coast Energy, Northstar Financial Group, and Summit Logistics.

TEAM GROWTH

Total headcount reached 352 employees at end of Q3:
- Engineering: 59 (added 5 this quarter)
- Sales & Marketing: 48
- Customer Success: 35
- Product: 18
- G&A: 22
- Executive: 8

Key hires: VP of Sales (Jennifer Walsh, ex-Datadog), Director of Customer Success (Robert Chen, ex-Twilio), and 3 senior engineers for the new AcmeInsight team.

OUTLOOK FOR Q4 2025

We project Q4 revenue of $7.8-8.2 million, with full-year 2025 revenue expected to reach $26-27 million. Key priorities for Q4 include:

1. AcmeGuard general availability launch
2. AcmeInsight beta program expansion to 50 customers
3. International expansion: opening London office in November 2025
4. SOC 2 Type II certification completion (expected December 2025)

The company is also beginning preliminary planning for a potential Series C funding round in H1 2026, targeting $80-100 million at a $400-500 million valuation.

RISK FACTORS

- Increasing competition from well-funded startups in the developer tools space
- Macroeconomic uncertainty affecting enterprise software purchasing decisions
- Key person dependency on founding team for strategic customer relationships
- Technical debt in legacy AcmeFlow v1/v2 codebase requiring ongoing maintenance resources
"""

def create_pdf():
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=letter)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("CustomTitle", parent=styles["Title"], fontSize=18, spaceAfter=20)
    heading_style = ParagraphStyle("CustomHeading", parent=styles["Heading2"], fontSize=13, spaceAfter=10, spaceBefore=16)
    body_style = ParagraphStyle("CustomBody", parent=styles["Normal"], fontSize=10, spaceAfter=8, leading=14)

    story = []
    for line in REPORT_TEXT.strip().split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 6))
        elif line.startswith("ACME CORP"):
            story.append(Paragraph(line, title_style))
        elif line.isupper() and len(line) > 5:
            story.append(Paragraph(line, heading_style))
        else:
            story.append(Paragraph(line, body_style))

    doc.build(story)
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    create_pdf()
