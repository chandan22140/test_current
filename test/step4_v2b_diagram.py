"""
SOARA Diagram Generator - Step 4: SOARA-V2b (Way1 Butterfly) Diagram
======================================================================

SOARA-V2b (Way1 with Butterfly): Butterfly Factorization
- Uses Butterfly structures (BOFT)
- Hierarchical block structure
- Recursive 2x2 blocks (butterfly factors)
- Visualized as sparsely connected blocks or recursive "X" patterns
"""

V2B_DIAGRAM = """<mxfile host="app.diagrams.net" modified="2026-02-08T16:41:00.000Z" agent="5.0" version="21.6.5">
  <diagram name="SOARA-V2b" id="soara-v2b">
    <mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="0" pageScale="1" pageWidth="850" pageHeight="1100">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        
        <!-- Title -->
        <mxCell id="v2b-title" value="&lt;b&gt;&lt;font style=&quot;font-size: 18px;&quot;&gt;SOARA-V2b (Way1)&lt;/font&gt;&lt;/b&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;" 
                vertex="1" parent="1">
          <mxGeometry x="1200" y="10" width="300" height="40" as="geometry" />
        </mxCell>
        
        <!-- Subtitle -->
        <mxCell id="v2b-subtitle" value="&lt;i&gt;Butterfly Factorization (BOFT)&lt;/i&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=12;" 
                vertex="1" parent="1">
          <mxGeometry x="1200" y="45" width="300" height="25" as="geometry" />
        </mxCell>
        
        <!-- M matrix representation title -->
        <mxCell id="v2b-m-title" value="&lt;b&gt;M Matrix (r × r)&lt;/b&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=14;" 
                vertex="1" parent="1">
          <mxGeometry x="1240" y="85" width="200" height="25" as="geometry" />
        </mxCell>
        
        <!-- Grid container for M matrix - 8x8 with Butterfly pattern -->
        <!-- Butterfly pattern often looks like recursive blocks -->
        <!-- Start with background gray -->
        
        <!-- Row 1 -->
        <mxCell id="v2b-cell-0-0" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="1250" y="120" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-0-1" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="1270" y="120" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-0-2" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1290" y="120" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-0-3" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1310" y="120" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-0-4" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="1330" y="120" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-0-5" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1350" y="120" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-0-6" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1370" y="120" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-0-7" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1390" y="120" width="20" height="20" as="geometry" />
        </mxCell>
        
        <!-- Row 2 -->
        <mxCell id="v2b-cell-1-0" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="1250" y="140" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-1-1" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="1270" y="140" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-1-2" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1290" y="140" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-1-3" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1310" y="140" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-1-4" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1330" y="140" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-1-5" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="1350" y="140" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-1-6" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1370" y="140" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-1-7" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1390" y="140" width="20" height="20" as="geometry" />
        </mxCell>
        
        <!-- Row 3 -->
        <mxCell id="v2b-cell-2-0" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1250" y="160" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-2-1" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1270" y="160" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-2-2" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="1290" y="160" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-2-3" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="1310" y="160" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-2-4" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1330" y="160" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-2-5" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1350" y="160" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-2-6" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="1370" y="160" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-2-7" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1390" y="160" width="20" height="20" as="geometry" />
        </mxCell>
        
        <!-- Row 4 -->
        <mxCell id="v2b-cell-3-0" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1250" y="180" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-3-1" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1270" y="180" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-3-2" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="1290" y="180" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-3-3" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="1310" y="180" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-3-4" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1330" y="180" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-3-5" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1350" y="180" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-3-6" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1370" y="180" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-3-7" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="1390" y="180" width="20" height="20" as="geometry" />
        </mxCell>
        
        <!-- Row 5 -->
        <mxCell id="v2b-cell-4-0" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="1250" y="200" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-4-1" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1270" y="200" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-4-2" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1290" y="200" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-4-3" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1310" y="200" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-4-4" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="1330" y="200" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-4-5" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="1350" y="200" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-4-6" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1370" y="200" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-4-7" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1390" y="200" width="20" height="20" as="geometry" />
        </mxCell>
        
        <!-- Row 6 -->
        <mxCell id="v2b-cell-5-0" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1250" y="220" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-5-1" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="1270" y="220" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-5-2" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1290" y="220" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-5-3" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1310" y="220" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-5-4" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="1330" y="220" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-5-5" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="1350" y="220" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-5-6" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1370" y="220" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-5-7" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1390" y="220" width="20" height="20" as="geometry" />
        </mxCell>
        
        <!-- Row 7 -->
        <mxCell id="v2b-cell-6-0" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1250" y="240" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-6-1" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1270" y="240" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-6-2" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="1290" y="240" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-6-3" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="1310" y="240" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-6-4" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1330" y="240" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-6-5" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1350" y="240" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-6-6" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="1370" y="240" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-6-7" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="1390" y="240" width="20" height="20" as="geometry" />
        </mxCell>
        
        <!-- Row 8 -->
        <mxCell id="v2b-cell-7-0" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1250" y="260" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-7-1" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1270" y="260" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-7-2" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1290" y="260" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-7-3" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1310" y="260" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-7-4" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1330" y="260" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-7-5" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#CCCCCC;strokeColor=#999999;" vertex="1" parent="1">
          <mxGeometry x="1350" y="260" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-7-6" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="1370" y="260" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v2b-cell-7-7" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="1390" y="260" width="20" height="20" as="geometry" />
        </mxCell>
        
        <!-- Description box -->
        <mxCell id="v2b-desc-box" value="" 
                style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F5E9;strokeColor=#388E3C;strokeWidth=2;" 
                vertex="1" parent="1">
          <mxGeometry x="1220" y="300" width="300" height="150" as="geometry" />
        </mxCell>
        
        <mxCell id="v2b-desc-title" value="&lt;b&gt;Properties:&lt;/b&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=top;fontSize=13;" 
                vertex="1" parent="1">
          <mxGeometry x="1230" y="310" width="100" height="25" as="geometry" />
        </mxCell>
        
        <mxCell id="v2b-desc1" value="• Hierarchical Butterfly structure (BOFT)" 
                style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=12;" 
                vertex="1" parent="1">
          <mxGeometry x="1235" y="335" width="270" height="20" as="geometry" />
        </mxCell>
        
        <mxCell id="v2b-desc2" value="• Recursive block-diagonal factors" 
                style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=12;" 
                vertex="1" parent="1">
          <mxGeometry x="1235" y="355" width="270" height="20" as="geometry" />
        </mxCell>
        
        <mxCell id="v2b-desc3" value="• Log-linear complexity O(r log r)" 
                style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=12;" 
                vertex="1" parent="1">
          <mxGeometry x="1235" y="375" width="270" height="20" as="geometry" />
        </mxCell>
        
        <mxCell id="v2b-desc4" value="• Composed of multiple butterfly layers" 
                style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=12;" 
                vertex="1" parent="1">
          <mxGeometry x="1235" y="395" width="270" height="20" as="geometry" />
        </mxCell>
        
        <mxCell id="v2b-desc5" value="• Sparse but global mixing" 
                style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=12;" 
                vertex="1" parent="1">
          <mxGeometry x="1235" y="415" width="270" height="20" as="geometry" />
        </mxCell>
        
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""

if __name__ == "__main__":
    with open("step4_v2b_diagram.xml", "w") as f:
        f.write(V2B_DIAGRAM)
    print("✓ Step 4: SOARA-V2b (War1 Butterfly) diagram generated!")
