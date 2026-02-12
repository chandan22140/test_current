import os
import re

def generate_final_diagram():
    """
    Generates the final SOARA draw.io diagram containing:
    1. General Architecture (Left) - UPDATED: U -> R_U -> Sigma -> R_V^T -> V^T
    2. SOARA-V1 (Way0) Variant (Right)
    3. SOARA-V2a (Way1 Givens) Variant (Right) - UPDATED: Specific Pairwise & Description
    4. SOARA-V2b (Way1 Butterfly) Variant (Right) - UPDATED: Checkerboard
    
    All merged into a single XML file.
    """
    
    # ==========================================================================================
    # 1. GENERAL ARCHITECTURE (LEFT)
    # ==========================================================================================
    
    gen_center_x = 170
    
    general_xml_content = f"""
        <!-- ========================================================================================== -->
        <!-- GENERAL ARCHITECTURE (LEFT) -->
        <!-- ========================================================================================== -->
        
        <!-- Title -->
        <mxCell id="gen-title" value="&lt;b&gt;&lt;font style=&quot;font-size: 18px;&quot;&gt;SOARA: General Architecture&lt;/font&gt;&lt;/b&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;" 
                vertex="1" parent="1">
          <mxGeometry x="20" y="20" width="300" height="40" as="geometry" />
        </mxCell>
        
        <!-- Subtitle with equation -->
        <mxCell id="gen-equation" value="&lt;i&gt;W = U ⊙ R&lt;sub&gt;U&lt;/sub&gt; ⊙ Σ ⊙ R&lt;sub&gt;V&lt;/sub&gt;&lt;sup&gt;T&lt;/sup&gt; ⊙ V&lt;sup&gt;T&lt;/sup&gt;&lt;/i&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=14;" 
                vertex="1" parent="1">
          <mxGeometry x="20" y="60" width="300" height="30" as="geometry" />
        </mxCell>
        
        <!-- output: h -->
        <mxCell id="gen-h-vector" value="" 
                style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFE6CC;strokeColor=#D79B00;strokeWidth=2;" 
                vertex="1" parent="1">
          <mxGeometry x="{gen_center_x - 70}" y="110" width="140" height="25" as="geometry" />
        </mxCell>
        <mxCell id="gen-h-label" value="&lt;b&gt;h&lt;/b&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;" 
                vertex="1" parent="1">
          <mxGeometry x="{gen_center_x + 80}" y="110" width="40" height="25" as="geometry" />
        </mxCell>
        <mxCell id="gen-arrow-h" value="" style="endArrow=classic;html=1;strokeWidth=1;" edge="1" parent="1">
             <mxGeometry width="50" height="50" relative="1" as="geometry">
                <mxPoint x="{gen_center_x}" y="150" as="sourcePoint" />
                <mxPoint x="{gen_center_x}" y="135" as="targetPoint" />
             </mxGeometry>
        </mxCell>

        <!-- U Matrix (Blue, Frozen) -->
        <mxCell id="gen-u-matrix" value="&lt;b&gt;U&lt;/b&gt;" 
                style="rounded=0;whiteSpace=wrap;html=1;fillColor=#6FA8DC;strokeColor=#0066CC;strokeWidth=2;fontSize=16;" 
                vertex="1" parent="1">
          <mxGeometry x="{gen_center_x - 60}" y="150" width="120" height="80" as="geometry" />
        </mxCell>
        
        <!-- R_U Matrix (Orange, Trainable) -->
        <mxCell id="gen-ru-matrix" value="&lt;b&gt;R&lt;sub&gt;U&lt;/sub&gt;&lt;/b&gt;" 
                style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;strokeWidth=2;fontSize=16;" 
                vertex="1" parent="1">
          <mxGeometry x="{gen_center_x - 40}" y="240" width="80" height="80" as="geometry" />
        </mxCell>
        
        <!-- Sigma (Diagonal, Gray/White) -->
        <mxCell id="gen-sigma-box" value="" 
                style="rounded=0;whiteSpace=wrap;html=1;fillColor=#E0E0E0;strokeColor=#666666;strokeWidth=2;" 
                vertex="1" parent="1">
          <mxGeometry x="{gen_center_x - 30}" y="330" width="60" height="60" as="geometry" />
        </mxCell>
        <mxCell id="gen-sigma-diag" value="" 
                style="endArrow=none;html=1;strokeColor=#FF9900;strokeWidth=3;" 
                edge="1" parent="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="{gen_center_x - 28}" y="388" as="sourcePoint" />
            <mxPoint x="{gen_center_x + 28}" y="332" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        <mxCell id="gen-sigma-label" value="&lt;b&gt;Σ&lt;/b&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=16;" 
                vertex="1" parent="1">
          <mxGeometry x="{gen_center_x + 35}" y="345" width="30" height="30" as="geometry" />
        </mxCell>

        <!-- R_V^T Matrix (Orange, Trainable) -->
        <mxCell id="gen-rv-matrix" value="&lt;b&gt;R&lt;sub&gt;V&lt;/sub&gt;&lt;sup&gt;T&lt;/sup&gt;&lt;/b&gt;" 
                style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;strokeWidth=2;fontSize=16;" 
                vertex="1" parent="1">
          <mxGeometry x="{gen_center_x - 40}" y="400" width="80" height="80" as="geometry" />
        </mxCell>

        <!-- V^T Matrix (Blue, Frozen) -->
        <mxCell id="gen-vt-matrix" value="&lt;b&gt;V&lt;sup&gt;T&lt;/sup&gt;&lt;/b&gt;" 
                style="rounded=0;whiteSpace=wrap;html=1;fillColor=#6FA8DC;strokeColor=#0066CC;strokeWidth=2;fontSize=16;" 
                vertex="1" parent="1">
          <mxGeometry x="{gen_center_x - 60}" y="490" width="120" height="60" as="geometry" />
        </mxCell>

        <!-- Input: x -->
        <mxCell id="gen-arrow-x" value="" style="endArrow=classic;html=1;strokeWidth=1;" edge="1" parent="1">
             <mxGeometry width="50" height="50" relative="1" as="geometry">
                <mxPoint x="{gen_center_x}" y="575" as="sourcePoint" />
                <mxPoint x="{gen_center_x}" y="555" as="targetPoint" />
             </mxGeometry>
        </mxCell>
        <mxCell id="gen-x-vector" value="" 
                style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFE6CC;strokeColor=#D79B00;strokeWidth=2;" 
                vertex="1" parent="1">
          <mxGeometry x="{gen_center_x - 70}" y="580" width="140" height="25" as="geometry" />
        </mxCell>
        <mxCell id="gen-x-label" value="&lt;b&gt;x&lt;/b&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;" 
                vertex="1" parent="1">
          <mxGeometry x="{gen_center_x + 80}" y="580" width="40" height="25" as="geometry" />
        </mxCell>
        
        <!-- Legend (Moved down) -->
        <mxCell id="gen-legend-box" value="" 
                style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F5F5F5;strokeColor=#666666;strokeWidth=1;dashed=1;" 
                vertex="1" parent="1">
          <mxGeometry x="20" y="630" width="200" height="90" as="geometry" />
        </mxCell>
        <mxCell id="gen-legend-title" value="&lt;b&gt;Legend:&lt;/b&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=top;" 
                vertex="1" parent="1">
          <mxGeometry x="30" y="635" width="60" height="20" as="geometry" />
        </mxCell>
        <mxCell id="gen-legend-frozen-box" value="" 
                style="rounded=0;whiteSpace=wrap;html=1;fillColor=#6FA8DC;strokeColor=#0066CC;strokeWidth=2;" 
                vertex="1" parent="1">
          <mxGeometry x="35" y="660" width="25" height="15" as="geometry" />
        </mxCell>
        <mxCell id="gen-legend-frozen-text" value="= frozen" 
                style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;" 
                vertex="1" parent="1">
          <mxGeometry x="65" y="657" width="80" height="20" as="geometry" />
        </mxCell>
        <mxCell id="gen-legend-train-box" value="" 
                style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;strokeWidth=2;" 
                vertex="1" parent="1">
          <mxGeometry x="35" y="682" width="25" height="15" as="geometry" />
        </mxCell>
        <mxCell id="gen-legend-train-text" value="= trainable" 
                style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;" 
                vertex="1" parent="1">
          <mxGeometry x="65" y="679" width="80" height="20" as="geometry" />
        </mxCell>
        <mxCell id="gen-legend-zero-box" value="" 
                style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;strokeWidth=2;" 
                vertex="1" parent="1">
          <mxGeometry x="35" y="704" width="25" height="15" as="geometry" />
        </mxCell>
        <mxCell id="gen-legend-zero-text" value="= zeros/identity" 
                style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;" 
                vertex="1" parent="1">
          <mxGeometry x="65" y="701" width="100" height="20" as="geometry" />
        </mxCell>
    """
    
    # ==========================================================================================
    # 2. V1 Diagram Part (x=400) - Unchanged
    # ==========================================================================================
    v1_start_x = 420
    v1_xml_content = """
        <!-- Title -->
        <mxCell id="v1-title" value="&lt;b&gt;&lt;font style=&quot;font-size: 16px;&quot;&gt;SOARA-V1 (Way0)&lt;/font&gt;&lt;/b&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;" 
                vertex="1" parent="1">
          <mxGeometry x="400" y="90" width="200" height="40" as="geometry" />
        </mxCell>
        <mxCell id="v1-m-title" value="&lt;b&gt;R Matrix&lt;/b&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=14;" 
                vertex="1" parent="1">
          <mxGeometry x="430" y="130" width="140" height="25" as="geometry" />
        </mxCell>
    """
    
    # Generate 8x8 grid for V1 (All Orange)
    grid_y_start = 160
    cell_size = 20
    for r in range(8):
        for c in range(8):
            color = "#FFB366" # Orange (Trainable)
            stroke = "#FF6600"
            v1_xml_content += f"""
        <mxCell id="v1-cell-{r}-{c}" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor={color};strokeColor={stroke};" vertex="1" parent="1">
          <mxGeometry x="{v1_start_x + c*cell_size}" y="{grid_y_start + r*cell_size}" width="{cell_size}" height="{cell_size}" as="geometry" />
        </mxCell>"""

    v1_xml_content += f"""
        <mxCell id="v1-desc-box" value="&lt;b&gt;Plain&lt;/b&gt;&lt;br&gt;All r² elements trainable&lt;br&gt;Orthogonality via reg." 
                style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF4E6;strokeColor=#FF9900;strokeWidth=2;align=center;verticalAlign=middle;fontSize=11;" 
                vertex="1" parent="1">
          <mxGeometry x="420" y="{grid_y_start + 8*cell_size + 10}" width="160" height="60" as="geometry" />
        </mxCell>
    """
    
    # ==========================================================================================
    # 3. V2a Diagram Part (x=650) - UPDATED TEXT DESCRIPTION
    # ==========================================================================================
    v2a_start_x = 650
    v2a_xml_content = """
        <mxCell id="v2a-title" value="&lt;b&gt;&lt;font style=&quot;font-size: 16px;&quot;&gt;SOARA-V2a (Way1)&lt;/font&gt;&lt;/b&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;" 
                vertex="1" parent="1">
          <mxGeometry x="630" y="90" width="200" height="40" as="geometry" />
        </mxCell>
        <mxCell id="v2a-m-title" value="&lt;b&gt;R Matrix&lt;/b&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=14;" 
                vertex="1" parent="1">
          <mxGeometry x="660" y="130" width="140" height="25" as="geometry" />
        </mxCell>
    """
    
    # Generate 8x8 grid for V2a (Specific Pairwise Logic)
    # Pairs: [(0, 1), (2, 7), (3, 6), (4, 5)]
    pairs = [(0, 1), (2, 7), (3, 6), (4, 5)]
    
    for r in range(8):
        for c in range(8):
            is_active = False
            
            # Check if (r, c) is part of any pair rotation
            # A pair (u, v) activates cells: (u,u), (u,v), (v,u), (v,v)
            for u, v in pairs:
                if (r == u and c == u) or \
                   (r == u and c == v) or \
                   (r == v and c == u) or \
                   (r == v and c == v):
                    is_active = True
                    break
            
            if is_active:
                color = "#FFB366" # Orange (Trainable)
                stroke = "#FF6600"
            else:
                color = "#CCCCCC" # Grey (Inactive)
                stroke = "#999999"
                
            v2a_xml_content += f"""
        <mxCell id="v2a-cell-{r}-{c}" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor={color};strokeColor={stroke};" vertex="1" parent="1">
          <mxGeometry x="{v2a_start_x + c*cell_size}" y="{grid_y_start + r*cell_size}" width="{cell_size}" height="{cell_size}" as="geometry" />
        </mxCell>"""

    # Updated description text
    v2a_xml_content += f"""
        <mxCell id="v2a-desc-box" value="&lt;b&gt;Parallel&lt;/b&gt;&lt;br&gt;Disjoint Givens&lt;br&gt;Trainable Pairs" 
                style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E6F4FF;strokeColor=#0066CC;strokeWidth=2;align=center;verticalAlign=middle;fontSize=11;" 
                vertex="1" parent="1">
          <mxGeometry x="650" y="{grid_y_start + 8*cell_size + 10}" width="160" height="60" as="geometry" />
        </mxCell>
    """

    # ==========================================================================================
    # 4. V2b Diagram Part (x=880) - CHECKERBOARD
    # ==========================================================================================
    v2b_start_x = 880
    v2b_xml_content = """
        <mxCell id="v2b-title" value="&lt;b&gt;&lt;font style=&quot;font-size: 16px;&quot;&gt;SOARA-V2b (Way1)&lt;/font&gt;&lt;/b&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;" 
                vertex="1" parent="1">
          <mxGeometry x="860" y="90" width="200" height="40" as="geometry" />
        </mxCell>
        <mxCell id="v2b-m-title" value="&lt;b&gt;R Matrix&lt;/b&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=14;" 
                vertex="1" parent="1">
          <mxGeometry x="890" y="130" width="140" height="25" as="geometry" />
        </mxCell>
    """
    
    # Generate 8x8 grid for V2b
    for r in range(8):
        for c in range(8):
            is_active = False
            
            # Check if within the two 4x4 diagonal blocks
            if (r // 4 == c // 4):
                # Within a 4x4 block, check for checkerboard pattern
                # Active cells: (0,0), (0,2), (1,1), (1,3)...
                # Logic: (r % 2) == (c % 2)
                if (r % 2) == (c % 2):
                    is_active = True
            
            if is_active:
                color = "#FFB366" # Orange (Trainable - originally black in user image)
                stroke = "#FF6600"
            else:
                color = "#CCCCCC" # Grey (Inactive)
                stroke = "#999999"
                
            v2b_xml_content += f"""
        <mxCell id="v2b-cell-{r}-{c}" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor={color};strokeColor={stroke};" vertex="1" parent="1">
          <mxGeometry x="{v2b_start_x + c*cell_size}" y="{grid_y_start + r*cell_size}" width="{cell_size}" height="{cell_size}" as="geometry" />
        </mxCell>"""

    v2b_xml_content += f"""
        <mxCell id="v2b-desc-box" value="&lt;b&gt;Butterfly&lt;/b&gt;&lt;br&gt;Recursive Block Structure&lt;br&gt;Two 4x4 checkerboards" 
                style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F5E9;strokeColor=#388E3C;strokeWidth=2;align=center;verticalAlign=middle;fontSize=11;" 
                vertex="1" parent="1">
          <mxGeometry x="880" y="{grid_y_start + 8*cell_size + 10}" width="160" height="60" as="geometry" />
        </mxCell>
    """

    # Combine everything
    final_xml = f"""<mxfile host="app.diagrams.net" modified="2026-02-08T18:30:00.000Z" agent="5.0" version="21.6.5">
  <diagram name="SOARA-Complete-v5" id="soara-complete-v5">
    <mxGraphModel dx="1400" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1150" pageHeight="850">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        {general_xml_content}
        {v1_xml_content}
        {v2a_xml_content}
        {v2b_xml_content}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""
    
    with open("soara_complete_diagram.xml", "w") as f:
        f.write(final_xml)
    print("✓ Successfully generated 'soara_complete_diagram.xml' with updated V2a description!")

if __name__ == "__main__":
    generate_final_diagram()
