"""
PDF Generator for AI Metadata Tagging Dashboard
Generates comprehensive PDF reports with all dashboard features
"""

import os
import tempfile
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.platypus.flowables import KeepTogether
from reportlab.platypus import Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import matplotlib
matplotlib.use('Agg')
import io
import base64

# Configure matplotlib for PDF generation
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

def generate_pdf_report(data):
    """
    Generate comprehensive PDF report from dashboard data
    """
    # Create temporary file for PDF
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    pdf_path = temp_dir / f"metadata_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    # Create PDF document
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )
    
    # Build PDF content
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#6366f1')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=15,
        spaceBefore=20,
        textColor=colors.HexColor('#111827')
    )
    
    # Title page
    story.append(Paragraph("AI Metadata Analysis Report", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
    story.append(Spacer(1, 30))
    
    try:
        # Extract data safely
        metadata = data.get('visualization_data', {}).get('metadata_tab', {})
        ad_insights = data.get('visualization_data', {}).get('ad_insights_tab', {})
        cross_val = data.get('cross_validation', {})
        
        # 1. Synopsis Section
        story.append(Paragraph("📝 Synopsis", heading_style))
        synopsis_data = metadata.get('synopsis_summary', {})
        if synopsis_data:
            synopsis_text = synopsis_data.get('synopsis', 'No synopsis available')
            word_count = synopsis_data.get('word_count', 0)
            duration = synopsis_data.get('duration', 0)
            
            story.append(Paragraph(synopsis_text, styles['Normal']))
            story.append(Spacer(1, 10))
            story.append(Paragraph(f"<b>Word Count:</b> {word_count:,} | <b>Duration:</b> {duration} minutes", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # 2. Lead Characters Section
        story.append(Paragraph("🎭 Lead Characters", heading_style))
        characters = metadata.get('lead_characters', [])
        
        if characters:
            # Characters table
            char_data = [['Character', 'Importance %', 'Mentions', 'Emotion']]
            for char in characters[:6]:  # Top 6 characters
                char_data.append([
                    char.get('name', 'Unknown'),
                    f"{char.get('importance', 0)}%",
                    str(char.get('mentions', 0)),
                    char.get('emotion', 'neutral').title()
                ])
            
            char_table = Table(char_data, colWidths=[2*inch, 1*inch, 1*inch, 1.2*inch])
            char_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366f1')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(char_table)
        
        story.append(Spacer(1, 20))
        
        # 3. Genre Classification
        story.append(Paragraph("🎬 Genre Classification", heading_style))
        genres = metadata.get('genre_classification', [])
        
        if genres:
            genre_data = [['Genre', 'Confidence']]
            for genre in genres:
                genre_data.append([
                    genre.get('genre', 'Unknown'),
                    f"{genre.get('confidence', 0)}%"
                ])
            
            genre_table = Table(genre_data, colWidths=[3*inch, 1.5*inch])
            genre_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34d399')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(genre_table)
        
        story.append(Spacer(1, 20))
        
        # 4. Keywords Analysis with Chart
        story.append(Paragraph("🔑 Top Keywords Analysis", heading_style))
        keywords_data = metadata.get('keywords_plot', {})
        
        if keywords_data.get('keywords'):
            # Create keywords chart
            try:
                chart_path = create_keywords_chart(keywords_data)
                if chart_path and os.path.exists(chart_path):
                    story.append(RLImage(chart_path, width=5*inch, height=3*inch))
                    story.append(Spacer(1, 10))
            except Exception as e:
                print(f"Warning: Could not create keywords chart: {e}")
                story.append(Paragraph("Keywords chart unavailable", styles['Normal']))
            
            # Keywords table
            keywords = keywords_data.get('keywords', [])[:10]
            percentages = keywords_data.get('percentages', [])[:10]
            
            kw_data = [['Keyword', 'Frequency %']]
            for i, keyword in enumerate(keywords):
                percentage = percentages[i] if i < len(percentages) else 0
                kw_data.append([keyword, f"{percentage}%"])
            
            kw_table = Table(kw_data, colWidths=[3*inch, 1.5*inch])
            kw_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#fbbf24')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(kw_table)
        
        story.append(PageBreak())
        
        # 5. Sentiment Analysis with Chart
        story.append(Paragraph("😊 Sentiment Analysis", heading_style))
        sentiment_data = metadata.get('sentiment_pie', {})
        
        if sentiment_data.get('labels'):
            # Create sentiment pie chart
            try:
                chart_path = create_sentiment_chart(sentiment_data)
                if chart_path and os.path.exists(chart_path):
                    story.append(RLImage(chart_path, width=4*inch, height=3*inch))
            except Exception as e:
                print(f"Warning: Could not create sentiment chart: {e}")
                story.append(Paragraph("Sentiment chart unavailable", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # 6. Named Entities Summary
        story.append(Paragraph("🧾 Named Entities Summary", heading_style))
        ner_data = metadata.get('named_entities', {})
        
        if ner_data:
            # Summary counts
            ner_summary = [
                ['Entity Type', 'Count'],
                ['People', str(ner_data.get('people_count', 0))],
                ['Locations', str(ner_data.get('locations_count', 0))],
                ['Organizations', str(ner_data.get('organizations_count', 0))],
                ['Total', str(ner_data.get('total', 0))]
            ]
            
            ner_table = Table(ner_summary, colWidths=[2*inch, 1*inch])
            ner_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8b5cf6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(ner_table)
            
            # Entity lists
            if ner_data.get('people'):
                story.append(Spacer(1, 15))
                story.append(Paragraph("<b>👤 People:</b> " + ", ".join(ner_data['people'][:10]), styles['Normal']))
            
            if ner_data.get('locations'):
                story.append(Paragraph("<b>📍 Locations:</b> " + ", ".join(ner_data['locations'][:10]), styles['Normal']))
                
            if ner_data.get('organizations'):
                story.append(Paragraph("<b>🏛 Organizations:</b> " + ", ".join(ner_data['organizations'][:5]), styles['Normal']))
        
        story.append(PageBreak())
        
        # 7. Advertisement Insights
        story.append(Paragraph("💡 Advertisement Insights", heading_style))
        
        if ad_insights:
            # Ad placement timeline chart
            timeline_data = ad_insights.get('ad_placement_timeline', {})
            if timeline_data.get('placements'):
                try:
                    chart_path = create_ad_timeline_chart(timeline_data)
                    if chart_path and os.path.exists(chart_path):
                        story.append(RLImage(chart_path, width=6*inch, height=3*inch))
                        story.append(Spacer(1, 15))
                except Exception as e:
                    print(f"Warning: Could not create ad timeline chart: {e}")
            
            # Ad recommendations table
            recommendations = ad_insights.get('ad_recommendations', [])
            if recommendations:
                story.append(Paragraph("🎯 Ad Placement Recommendations", heading_style))
                
                rec_data = [['Scene', 'Suitability', 'Ad Types', 'Reasoning']]
                for rec in recommendations[:5]:  # Top 5 recommendations
                    scene = wrap_text_for_cell(rec.get('scene', ''), 40)
                    suitability = f"{rec.get('suitability', 0)}%"
                    ad_types = wrap_text_for_cell(", ".join(rec.get('ad_types', [])[:2]), 20)
                    reasoning = wrap_text_for_cell(rec.get('reasoning', ''), 50)
                    
                    rec_data.append([scene, suitability, ad_types, reasoning])
                
                rec_table = Table(rec_data, colWidths=[1.8*inch, 0.7*inch, 1.3*inch, 2*inch])
                rec_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f87171')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP')
                ]))
                story.append(rec_table)
        
        story.append(PageBreak())
        
        # 8. Evaluation & Validation Metrics
        story.append(Paragraph("📊 Evaluation & Validation", heading_style))
        
        if cross_val:
            # Confidence scores
            confidence = cross_val.get('confidence_scores', {})
            if confidence:
                story.append(Paragraph("🎯 Confidence Metrics", heading_style))
                
                conf_data = [
                    ['Metric', 'Score'],
                    ['LLM Confidence', f"{confidence.get('llm_confidence', 0)*100:.1f}%"],
                    ['NLP Confidence', f"{confidence.get('nlp_confidence', 0)*100:.1f}%"],
                    ['Agreement Score', f"{confidence.get('agreement_score', 0)*100:.1f}%"],
                    ['Overall Confidence', f"{confidence.get('overall_confidence', 0)*100:.1f}%"]
                ]
                
                conf_table = Table(conf_data, colWidths=[2.5*inch, 1.5*inch])
                conf_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366f1')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(conf_table)
                story.append(Spacer(1, 20))
            
            # Performance metrics
            performance = cross_val.get('performance_metrics', {})
            if performance:
                story.append(Paragraph("⚡ Performance Metrics", heading_style))
                
                speed = performance.get('processing_speed', {})
                accuracy = performance.get('accuracy_metrics', {})
                
                perf_data = [['Metric', 'LLM', 'NLP']]
                
                if speed:
                    perf_data.append(['Processing Time', f"{speed.get('llm_time', 0)}s", f"{speed.get('nlp_time', 0):.1f}s"])
                
                if accuracy:
                    perf_data.append(['Sentiment Accuracy', f"{accuracy.get('sentiment_accuracy', {}).get('llm', 0)}%", f"{accuracy.get('sentiment_accuracy', {}).get('nlp', 0)}%"])
                    perf_data.append(['Entity Accuracy', f"{accuracy.get('entity_extraction_accuracy', {}).get('llm', 0)}%", f"{accuracy.get('entity_extraction_accuracy', {}).get('nlp', 0)}%"])
                
                perf_table = Table(perf_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
                perf_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34d399')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(perf_table)
        
        # Add charts to story
        add_charts_to_story(story, metadata, ad_insights, cross_val)
        
        # Build PDF with error handling
        try:
            doc.build(story)
            print(f"PDF generated successfully: {pdf_path}")
        except Exception as build_error:
            print(f"Error building PDF: {build_error}")
            # Create minimal PDF on build error
            minimal_story = [
                Paragraph("PDF Generation Error", title_style),
                Paragraph(f"Build Error: {str(build_error)}", styles['Normal'])
            ]
            doc.build(minimal_story)
        
        return str(pdf_path)
        
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        # Create error PDF
        error_story = [
            Paragraph("PDF Generation Error", title_style),
            Paragraph(f"Error: {str(e)}", styles['Normal']),
            Paragraph("Please try again or contact support.", styles['Normal'])
        ]
        doc.build(error_story)
        return str(pdf_path)
    
def safe_text_truncate(text, max_length=50):
    """Safely truncate text to prevent overflow"""
    if not text:
        return ""
    text = str(text)
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def wrap_text_for_cell(text, max_width=30):
    """Wrap text for table cells to prevent overflow"""
    if not text:
        return ""
    text = str(text)
    if len(text) <= max_width:
        return text
    
    # Split into words and wrap
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= max_width:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return '<br/>'.join(lines)
    
def add_charts_to_story(story, metadata, ad_insights, cross_val):
    """Add matplotlib charts to PDF story"""
    
    # Create charts and add to story
    chart_paths = []
    
    try:
        # Keywords chart
        keywords_data = metadata.get('keywords_plot', {})
        if keywords_data.get('keywords'):
            kw_chart = create_keywords_chart(keywords_data)
            if kw_chart:
                chart_paths.append(kw_chart)
        
        # Sentiment chart  
        sentiment_data = metadata.get('sentiment_pie', {})
        if sentiment_data.get('labels'):
            sent_chart = create_sentiment_chart(sentiment_data)
            if sent_chart:
                chart_paths.append(sent_chart)
        
        # Emotions chart
        emotions_data = metadata.get('emotion_pie', {})
        if emotions_data.get('labels'):
            emo_chart = create_emotions_chart(emotions_data)
            if emo_chart:
                chart_paths.append(emo_chart)
        
        # Ad timeline chart
        if ad_insights:
            timeline_data = ad_insights.get('ad_placement_timeline', {})
            if timeline_data.get('placements'):
                ad_chart = create_ad_timeline_chart(timeline_data)
                if ad_chart:
                    chart_paths.append(ad_chart)
        
        # Confidence gauge
        if cross_val:
            confidence = cross_val.get('validation_summary', {}).get('overall_reliability', 50)
            conf_chart = create_confidence_gauge(confidence)
            if conf_chart:
                chart_paths.append(conf_chart)
    
    except Exception as e:
        print(f"Error creating charts: {str(e)}")
    
    return chart_paths

def create_keywords_chart(keywords_data):
    """Create keywords bar chart"""
    try:
        keywords = keywords_data.get('keywords', [])[:8]  # Top 8
        percentages = keywords_data.get('percentages', [])[:8]
        
        if not keywords or not percentages:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 5))
        y_pos = np.arange(len(keywords))
        
        bars = ax.barh(y_pos, percentages, color='#6366f1', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(keywords)
        ax.set_xlabel('Frequency (%)')
        ax.set_title('Top Keywords Distribution', fontsize=14, fontweight='bold', color='#111827')
        
        # Add value labels on bars
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                   f'{pct}%', ha='left', va='center', fontweight='bold')
        
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Save chart
        chart_path = f"temp/keywords_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return chart_path
        
    except Exception as e:
        print(f"Error creating keywords chart: {str(e)}")
        return None

def create_sentiment_chart(sentiment_data):
    """Create sentiment pie chart"""
    try:
        labels = sentiment_data.get('labels', [])
        values = sentiment_data.get('values', [])
        
        if not labels or not values:
            return None
        
        fig, ax = plt.subplots(figsize=(6, 6))
        colors_map = ['#34d399', '#f87171', '#fbbf24']  # Green, Red, Yellow
        
        wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', 
                                         colors=colors_map[:len(labels)], startangle=90)
        
        ax.set_title('Sentiment Distribution', fontsize=14, fontweight='bold', color='#111827', pad=20)
        
        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        # Save chart
        chart_path = f"temp/sentiment_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return chart_path
        
    except Exception as e:
        print(f"Error creating sentiment chart: {str(e)}")
        return None

def create_emotions_chart(emotions_data):
    """Create emotions pie chart"""
    try:
        labels = emotions_data.get('labels', [])
        values = emotions_data.get('values', [])
        
        if not labels or not values:
            return None
        
        fig, ax = plt.subplots(figsize=(6, 6))
        colors_map = ['#60a5fa', '#f87171', '#9ca3af', '#fbbf24', '#8b5cf6']
        
        wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.0f', 
                                         colors=colors_map[:len(labels)], startangle=45)
        
        ax.set_title('Character Emotions Distribution', fontsize=14, fontweight='bold', color='#111827', pad=20)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        chart_path = f"temp/emotions_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return chart_path
        
    except Exception as e:
        print(f"Error creating emotions chart: {str(e)}")
        return None

def create_ad_timeline_chart(timeline_data):
    """Create ad placement timeline chart"""
    try:
        placements = timeline_data.get('placements', [])
        
        if not placements:
            return None
        
        timestamps = [p.get('timestamp', '') for p in placements]
        suitability = [p.get('suitability', 0) for p in placements]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        ax.plot(range(len(timestamps)), suitability, marker='o', linewidth=2, 
               markersize=6, color='#6366f1', markerfacecolor='#8b5cf6')
        ax.fill_between(range(len(timestamps)), suitability, alpha=0.3, color='#6366f1')
        
        ax.set_xticks(range(len(timestamps)))
        ax.set_xticklabels(timestamps, rotation=45, ha='right')
        ax.set_ylabel('Suitability Score (%)')
        ax.set_title('Ad Placement Timeline', fontsize=14, fontweight='bold', color='#111827')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        
        chart_path = f"temp/ad_timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return chart_path
        
    except Exception as e:
        print(f"Error creating ad timeline chart: {str(e)}")
        return None

def create_confidence_gauge(reliability_score):
    """Create confidence gauge chart"""
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        r1, r2 = 0.7, 1.0
        
        # Background arc
        ax.fill_between(theta, r1, r2, color='#e5e7eb', alpha=0.3)
        
        # Confidence arc
        confidence_theta = theta[:int(reliability_score)]
        color = '#34d399' if reliability_score >= 70 else '#fbbf24' if reliability_score >= 40 else '#f87171'
        ax.fill_between(confidence_theta, r1, r2, color=color, alpha=0.8)
        
        # Add text
        ax.text(np.pi/2, 0.4, f'{reliability_score:.1f}%', 
               ha='center', va='center', fontsize=20, fontweight='bold', color='#111827')
        ax.text(np.pi/2, 0.2, 'Overall Reliability', 
               ha='center', va='center', fontsize=12, color='#6b7280')
        
        ax.set_xlim(0, np.pi)
        ax.set_ylim(0, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Confidence Assessment', fontsize=14, fontweight='bold', color='#111827', pad=20)
        
        plt.tight_layout()
        
        chart_path = f"temp/confidence_gauge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return chart_path
        
    except Exception as e:
        print(f"Error creating confidence gauge: {str(e)}")
        return None

def cleanup_temp_files():
    """Clean up temporary chart files"""
    try:
        temp_dir = Path("temp")
        if temp_dir.exists():
            for file in temp_dir.glob("*.png"):
                try:
                    file.unlink()
                except:
                    pass  # Ignore errors during cleanup
    except Exception as e:
        print(f"Warning: Could not clean up temp files: {str(e)}")

# Cleanup temp files on module import
import atexit
atexit.register(cleanup_temp_files)