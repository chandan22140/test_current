"""
SOARA Diagram Generator - Step 2: SOARA-V1 (Way0) Diagram
===========================================================

SOARA-V1 (Way0): Direct optimization with orthogonality regularization
- M is a full dense matrix, directly optimized
- Uses regularization to enforce orthogonality: ||M^T M - I||_F
- All elements of M are trainable (orange)
"""

V1_DIAGRAM = """<mxfile host="app.diagrams.net" modified="2026-02-08T16:41:00.000Z" agent="5.0" version="21.6.5">
  <diagram name="SOARA-V1" id="soara-v1">
    <mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="0" pageScale="1" pageWidth="850" pageHeight="1100">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        
        <!-- Title -->
        <mxCell id="v1-title" value="&lt;b&gt;&lt;font style=&quot;font-size: 18px;&quot;&gt;SOARA-V1 (Way0)&lt;/font&gt;&lt;/b&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;" 
                vertex="1" parent="1">
          <mxGeometry x="400" y="10" width="300" height="40" as="geometry" />
        </mxCell>
        
        <!-- Subtitle -->
        <mxCell id="v1-subtitle" value="&lt;i&gt;Direct Optimization with Regularization&lt;/i&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=12;" 
                vertex="1" parent="1">
          <mxGeometry x="400" y="45" width="300" height="25" as="geometry" />
        </mxCell>
        
        <!-- M matrix representation title -->
        <mxCell id="v1-m-title" value="&lt;b&gt;M Matrix (r × r)&lt;/b&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=14;" 
                vertex="1" parent="1">
          <mxGeometry x="440" y="85" width="200" height="25" as="geometry" />
        </mxCell>
        
        <!-- Grid container for M matrix - 8x8 grid visualization -->
        <!-- Row 1 -->
        <mxCell id="v1-cell-0-0" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="450" y="120" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-0-1" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="470" y="120" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-0-2" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="490" y="120" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-0-3" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="510" y="120" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-0-4" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="530" y="120" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-0-5" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="550" y="120" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-0-6" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="570" y="120" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-0-7" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="590" y="120" width="20" height="20" as="geometry" />
        </mxCell>
        
        <!-- Row 2 -->
        <mxCell id="v1-cell-1-0" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="450" y="140" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-1-1" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="470" y="140" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-1-2" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="490" y="140" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-1-3" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="510" y="140" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-1-4" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="530" y="140" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-1-5" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="550" y="140" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-1-6" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="570" y="140" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-1-7" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="590" y="140" width="20" height="20" as="geometry" />
        </mxCell>
        
        <!-- Row 3 -->
        <mxCell id="v1-cell-2-0" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="450" y="160" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-2-1" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="470" y="160" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-2-2" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="490" y="160" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-2-3" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="510" y="160" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-2-4" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="530" y="160" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-2-5" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="550" y="160" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-2-6" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="570" y="160" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-2-7" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="590" y="160" width="20" height="20" as="geometry" />
        </mxCell>
        
        <!-- Row 4 -->
        <mxCell id="v1-cell-3-0" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="450" y="180" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-3-1" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="470" y="180" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-3-2" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="490" y="180" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-3-3" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="510" y="180" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-3-4" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="530" y="180" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-3-5" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="550" y="180" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-3-6" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="570" y="180" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-3-7" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="590" y="180" width="20" height="20" as="geometry" />
        </mxCell>
        
        <!-- Row 5 -->
        <mxCell id="v1-cell-4-0" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="450" y="200" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-4-1" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="470" y="200" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-4-2" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="490" y="200" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-4-3" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="510" y="200" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-4-4" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="530" y="200" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-4-5" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="550" y="200" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-4-6" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="570" y="200" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-4-7" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="590" y="200" width="20" height="20" as="geometry" />
        </mxCell>
        
        <!-- Row 6 -->
        <mxCell id="v1-cell-5-0" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="450" y="220" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-5-1" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="470" y="220" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-5-2" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="490" y="220" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-5-3" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="510" y="220" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-5-4" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="530" y="220" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-5-5" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="550" y="220" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-5-6" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="570" y="220" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-5-7" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="590" y="220" width="20" height="20" as="geometry" />
        </mxCell>
        
        <!-- Row 7 -->
        <mxCell id="v1-cell-6-0" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="450" y="240" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-6-1" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="470" y="240" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-6-2" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="490" y="240" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-6-3" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="510" y="240" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-6-4" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="530" y="240" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-6-5" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="550" y="240" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-6-6" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="570" y="240" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-6-7" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="590" y="240" width="20" height="20" as="geometry" />
        </mxCell>
        
        <!-- Row 8 -->
        <mxCell id="v1-cell-7-0" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="450" y="260" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-7-1" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="470" y="260" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-7-2" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="490" y="260" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-7-3" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="510" y="260" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-7-4" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="530" y="260" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-7-5" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="550" y="260" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-7-6" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="570" y="260" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="v1-cell-7-7" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;" vertex="1" parent="1">
          <mxGeometry x="590" y="260" width="20" height="20" as="geometry" />
        </mxCell>
        
        <!-- Description box -->
        <mxCell id="v1-desc-box" value="" 
                style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF4E6;strokeColor=#FF9900;strokeWidth=2;" 
                vertex="1" parent="1">
          <mxGeometry x="420" y="300" width="300" height="130" as="geometry" />
        </mxCell>
        
        <mxCell id="v1-desc-title" value="&lt;b&gt;Properties:&lt;/b&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=top;fontSize=13;" 
                vertex="1" parent="1">
          <mxGeometry x="430" y="310" width="100" height="25" as="geometry" />
        </mxCell>
        
        <mxCell id="v1-desc1" value="• All r² elements trainable" 
                style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=12;" 
                vertex="1" parent="1">
          <mxGeometry x="435" y="335" width="270" height="20" as="geometry" />
        </mxCell>
        
        <mxCell id="v1-desc2" value="• Enforces orthogonality via regularization:" 
                style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=12;" 
                vertex="1" parent="1">
          <mxGeometry x="435" y="355" width="270" height="20" as="geometry" />
        </mxCell>
        
        <mxCell id="v1-desc3" value="  𝓛&lt;sub&gt;reg&lt;/sub&gt; = λ ||M&lt;sup&gt;T&lt;/sup&gt;M - I||&lt;sub&gt;F&lt;/sub&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=12;" 
                vertex="1" parent="1">
          <mxGeometry x="445" y="375" width="260" height="20" as="geometry" />
        </mxCell>
        
        <mxCell id="v1-desc4" value="• Parameters: O(r²)" 
                style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=12;" 
                vertex="1" parent="1">
          <mxGeometry x="435" y="400" width="270" height="20" as="geometry" />
        </mxCell>
        
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""

if __name__ == "__main__":
    with open("step2_v1_diagram.xml", "w") as f:
        f.write(V1_DIAGRAM)
    print("✓ Step 2: SOARA-V1 (Way0) diagram generated!")
