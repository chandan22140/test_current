"""
SOARA Diagram Generator - Step 3: SOARA-V2a (Way1 with Givens) Diagram
========================================================================

SOARA-V2a (Way1 with Givens): Greedy sequential Givens rotations
- M is composed of sequential Givens rotation layers
- Each layer has disjoint Givens rotation pairs
- Banded structure due to sequential training
- Orange cells represent trainable rotation angles
- Gray cells represent identity (no rotation)
"""

V2A_DIAGRAM = """<mxfile host="app.diagrams.net" modified="2026-02-08T16:41:00.000Z" agent="5.0" version="21.6.5">
  <diagram name="SOARA-V2a" id="soara-v2a">
    <mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="0" pageScale="1" pageWidth="850" pageHeight="1100">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        
        <!-- Title -->
        <mxCell id="v2a-title" value="&lt;b&gt;&lt;font style=&quot;font-size: 18px;&quot;&gt;SOARA-V2a (Way1)&lt;/font&gt;&lt;/b&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;" 
                vertex="1" parent="1">
          <mxGeometry x="800" y="10" width="300" height="40" as="geometry" />
        </mxCell>
        
        <!-- Subtitle -->
        <mxCell id="v2a-subtitle" value="&lt;i&gt;Sequential Givens Rotations (Banded)&lt;/i&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=12;" 
                vertex="1" parent="1">
          <mxGeometry x="800" y="45" width="300" height="25" as="geometry" />
        </mxCell>
        
        <!-- M matrix representation title -->
        <mxCell id="v2a-m-title" value="&lt;b&gt;M Matrix (r × r)&lt;/b&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=14;" 
                vertex="1" parent="1">
          <mxGeometry x="840" y="85" width="200" height="25" as="geometry" />
        </mxCell>
        
        <!-- Grid container for M matrix - 8x8 with banded pattern -->
        <!-- Banded pattern: trainable diagonal band, rest is identity/gray -->
        <!-- Row 1: diagonal + neighbor bands -->
        <mxCell id="v2a-cell-0-0" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="850" y="120" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-0-1" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="870" y="120" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-0-2" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="890" y="120" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-0-3" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="910" y="120" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-0-4" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="930" y="120" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-0-5" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="950" y="120" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-0-6" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="970" y="120" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-0-7" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="990" y="120" width="20" height="20" as="geometry" />
        </mxCell>
        
        <!-- Row 2 -->
        <mxCell id="v2a-cell-1-0" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="850" y="140" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-1-1" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="870" y="140" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-1-2" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="890" y="140" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-1-3" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="910" y="140" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-1-4" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="930" y="140" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-1-5" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="950" y="140" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-1-6" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="970" y="140" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-1-7" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="990" y="140" width="20" height="20" as="geometry" />
        </mxCell>
        
        <!-- Row 3 -->
        <mxCell id="v2a-cell-2-0" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="850" y="160" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-2-1" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="870" y="160" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-2-2" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="890" y="160" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-2-3" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="910" y="160" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-2-4" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="930" y="160" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-2-5" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="950" y="160" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-2-6" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="970" y="160" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-2-7" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="990" y="160" width="20" height="20" as="geometry" />
        </mxCell>
        
        <!-- Row 4 -->
        <mxCell id="v2a-cell-3-0" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="850" y="180" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-3-1" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="870" y="180" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-3-2" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="890" y="180" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-3-3" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="910" y="180" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-3-4" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="930" y="180" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-3-5" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="950" y="180" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-3-6" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="970" y="180" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-3-7" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="990" y="180" width="20" height="20" as="geometry" />
        </mxCell>
        
        <!-- Row 5 -->
        <mxCell id="v2a-cell-4-0" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="850" y="200" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-4-1" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="870" y="200" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-4-2" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="890" y="200" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-4-3" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="910" y="200" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-4-4" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="930" y="200" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-4-5" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="950" y="200" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-4-6" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="970" y="200" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-4-7" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="990" y="200" width="20" height="20" as="geometry" />
        </mxCell>
        
        <!-- Row 6 -->
        <mxCell id="v2a-cell-5-0" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="850" y="220" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-5-1" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="870" y="220" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-5-2" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="890" y="220" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-5-3" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="910" y="220" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-5-4" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="930" y="220" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-5-5" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="950" y="220" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-5-6" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="970" y="220" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-5-7" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="990" y="220" width="20" height="20" as="geometry" />
        </mxCell>
        
        <!-- Row 7 -->
        <mxCell id="v2a-cell-6-0" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="850" y="240" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-6-1" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="870" y="240" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-6-2" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="890" y="240" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-6-3" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="910" y="240" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-6-4" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="930" y="240" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-6-5" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="950" y="240" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-6-6" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="970" y="240" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-6-7" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="990" y="240" width="20" height="20" as="geometry" />
        </mxCell>
        
        <!-- Row 8 -->
        <mxCell id="v2a-cell-7-0" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="850" y="260" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-7-1" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="870" y="260" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-7-2" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="890" y="260" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-7-3" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="910" y="260" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-7-4" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="930" y="260" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-7-5" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="950" y="260" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-7-6" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="970" y="260" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2a-cell-7-7" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="990" y="260" width="20" height="20" as="geometry" />
        </mxCell>
        
        <!-- Description box -->
        <mxCell id="v2a-desc-box" value="" 
                style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E6F4FF;strokeColor=#0066CC;strokeWidth=2;" 
                vertex="1" parent="1">
          <mxGeometry x="820" y="300" width="300" height="150" as="geometry" />
        </mxCell>
        
        <mxCell id="v2a-desc-title" value="&lt;b&gt;Properties:&lt;/b&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=top;fontSize=13;" 
                vertex="1" parent="1">
          <mxGeometry x="830" y="310" width="100" height="25" as="geometry" />
        </mxCell>
        
        <mxCell id="v2a-desc1" value="• Banded structure from sequential training" 
                style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=12;" 
                vertex="1" parent="1">
          <mxGeometry x="835" y="335" width="270" height="20" as="geometry" />
        </mxCell>
        
        <mxCell id="v2a-desc2" value="• Composed of Givens rotation layers" 
                style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=12;" 
                vertex="1" parent="1">
          <mxGeometry x="835" y="355" width="270" height="20" as="geometry" />
        </mxCell>
        
        <mxCell id="v2a-desc3" value="• Each layer: disjoint (i, j) rotation pairs" 
                style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=12;" 
                vertex="1" parent="1">
          <mxGeometry x="835" y="375" width="270" height="20" as="geometry" />
        </mxCell>
        
        <mxCell id="v2a-desc4" value="• Trainable: rotation angles θ&lt;sub&gt;ij&lt;/sub&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=12;" 
                vertex="1" parent="1">
          <mxGeometry x="835" y="395" width="270" height="20" as="geometry" />
        </mxCell>
        
        <mxCell id="v2a-desc5" value="• Parameters: O(r) per layer" 
                style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=12;" 
                vertex="1" parent="1">
          <mxGeometry x="835" y="415" width="270" height="20" as="geometry" />
        </mxCell>
        
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""

if __name__ == "__main__":
    with open("step3_v2a_diagram.xml", "w") as f:
        f.write(V2A_DIAGRAM)
    print("✓ Step 3: SOARA-V2a (Way1 Givens) diagram generated!")
