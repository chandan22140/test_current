"""
SOARA Diagram Generator - Step 1: General SOARA Diagram
=========================================================

This diagram shows the general architecture of SOARA:
- Input vectors (h, x) with dimensions
- Matrices: U (frozen), Σ (diagonal), R (rotation, trainable), V^T (frozen)  
- Mathematical decomposition: W = U @ R_U @ Σ @ R_V^T @ V^T
"""

# This is the draw.io XML for the GENERAL SOARA diagram

GENERAL_DIAGRAM = """<mxfile host="app.diagrams.net" modified="2026-02-08T16:41:00.000Z" agent="5.0" version="21.6.5">
  <diagram name="SOARA-General" id="soara-general">
    <mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="0" pageScale="1" pageWidth="850" pageHeight="1100">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        
        <!-- Title -->
        <mxCell id="gen-title" value="&lt;b&gt;&lt;font style=&quot;font-size: 18px;&quot;&gt;SOARA: General Architecture&lt;/font&gt;&lt;/b&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;" 
                vertex="1" parent="1">
          <mxGeometry x="10" y="10" width="350" height="40" as="geometry" />
        </mxCell>
        
        <!-- Subtitle with equation -->
        <mxCell id="gen-equation" value="&lt;i&gt;W = U ⊙ R&lt;sub&gt;U&lt;/sub&gt; ⊙ Σ ⊙ R&lt;sub&gt;V&lt;/sub&gt;&lt;sup&gt;T&lt;/sup&gt; ⊙ V&lt;sup&gt;T&lt;/sup&gt;&lt;/i&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=14;" 
                vertex="1" parent="1">
          <mxGeometry x="10" y="50" width="350" height="30" as="geometry" />
        </mxCell>
        
        <!-- Left side: dim labels -->
        <mxCell id="gen-dimout-label" value="&lt;i&gt;dim&lt;sub&gt;out&lt;/sub&gt;&lt;/i&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;" 
                vertex="1" parent="1">
          <mxGeometry x="20" y="95" width="60" height="25" as="geometry" />
        </mxCell>
        
        <mxCell id="gen-h-vector" value="" 
                style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFE6CC;strokeColor=#D79B00;strokeWidth=2;" 
                vertex="1" parent="1">
          <mxGeometry x="95" y="95" width="140" height="25" as="geometry" />
        </mxCell>
        
        <mxCell id="gen-h-label" value="&lt;b&gt;h&lt;/b&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;" 
                vertex="1" parent="1">
          <mxGeometry x="145" y="95" width="40" height="25" as="geometry" />
        </mxCell>
        
        <!-- U matrix (frozen, blue) -->
        <mxCell id="gen-u-matrix" value="" 
                style="rounded=0;whiteSpace=wrap;html=1;fillColor=#6FA8DC;strokeColor=#0066CC;strokeWidth=2;" 
                vertex="1" parent="1">
          <mxGeometry x="95" y="135" width="90" height="120" as="geometry" />
        </mxCell>
        
        <mxCell id="gen-u-label" value="&lt;b&gt;&lt;font style=&quot;font-size: 16px;&quot;&gt;U&lt;/font&gt;&lt;/b&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;" 
                vertex="1" parent="1">
          <mxGeometry x="120" y="182" width="40" height="30" as="geometry" />
        </mxCell>
        
        <!-- Sigma matrix (diagonal, gray) -->
        <mxCell id="gen-sigma-box" value="" 
                style="rounded=0;whiteSpace=wrap;html=1;fillColor=#E0E0E0;strokeColor=#666666;strokeWidth=2;" 
                vertex="1" parent="1">
          <mxGeometry x="100" y="170" width="60" height="60" as="geometry" />
        </mxCell>
        
        <!-- Diagonal line for Sigma -->
        <mxCell id="gen-sigma-diag" value="" 
                style="endArrow=none;html=1;strokeColor=#FF9900;strokeWidth=3;" 
                edge="1" parent="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="103" y="227" as="sourcePoint" />
            <mxPoint x="157" y="173" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        
        <mxCell id="gen-sigma-label" value="&lt;b&gt;Σ&lt;/b&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=14;" 
                vertex="1" parent="1">
          <mxGeometry x="110" y="188" width="40" height="25" as="geometry" />
        </mxCell>
        
        <!-- M matrix (trainable rotation, orange) -->
        <mxCell id="gen-m-matrix" value="" 
                style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;strokeWidth=2;" 
                vertex="1" parent="1">
          <mxGeometry x="145" y="150" width="70" height="70" as="geometry" />
        </mxCell>
        
        <mxCell id="gen-m-label" value="&lt;b&gt;&lt;font style=&quot;font-size: 16px;&quot;&gt;M&lt;/font&gt;&lt;/b&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;" 
                vertex="1" parent="1">
          <mxGeometry x="160" y="172" width="40" height="30" as="geometry" />
        </mxCell>
        
        <!-- V^T matrix (frozen, blue) -->
        <mxCell id="gen-vt-matrix" value="" 
                style="rounded=0;whiteSpace=wrap;html=1;fillColor=#6FA8DC;strokeColor=#0066CC;strokeWidth=2;" 
                vertex="1" parent="1">
          <mxGeometry x="95" y="270" width="140" height="60" as="geometry" />
        </mxCell>
        
        <mxCell id="gen-vt-label" value="&lt;b&gt;&lt;font style=&quot;font-size: 16px;&quot;&gt;V&lt;sup&gt;T&lt;/sup&gt;&lt;/font&gt;&lt;/b&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;" 
                vertex="1" parent="1">
          <mxGeometry x="145" y="285" width="40" height="30" as="geometry" />
        </mxCell>
        
        <!-- x vector -->
        <mxCell id="gen-x-vector" value="" 
                style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFE6CC;strokeColor=#D79B00;strokeWidth=2;" 
                vertex="1" parent="1">
          <mxGeometry x="95" y="345" width="140" height="25" as="geometry" />
        </mxCell>
        
        <mxCell id="gen-x-label" value="&lt;b&gt;x&lt;/b&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;" 
                vertex="1" parent="1">
          <mxGeometry x="145" y="345" width="40" height="25" as="geometry" />
        </mxCell>
        
        <mxCell id="gen-dimin-label" value="&lt;i&gt;dim&lt;sub&gt;in&lt;/sub&gt;&lt;/i&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;" 
                vertex="1" parent="1">
          <mxGeometry x="20" y="345" width="60" height="25" as="geometry" />
        </mxCell>
        
        <!-- Arrow to instantiations -->
        <mxCell id="gen-arrow" value="" 
                style="endArrow=classic;html=1;strokeWidth=3;strokeColor=#000000;" 
                edge="1" parent="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="230" y="185" as="sourcePoint" />
            <mxPoint x="280" y="185" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        
        <mxCell id="gen-arrow-label" value="&lt;i&gt;Different M&lt;br/&gt;implementations&lt;/i&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=11;" 
                vertex="1" parent="1">
          <mxGeometry x="220" y="145" width="90" height="35" as="geometry" />
        </mxCell>
        
        <!-- Legend box -->
        <mxCell id="gen-legend-box" value="" 
                style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F5F5F5;strokeColor=#666666;strokeWidth=1;dashed=1;" 
                vertex="1" parent="1">
          <mxGeometry x="20" y="400" width="210" height="90" as="geometry" />
        </mxCell>
        
        <mxCell id="gen-legend-title" value="&lt;b&gt;Legend:&lt;/b&gt;" 
                style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=top;" 
                vertex="1" parent="1">
          <mxGeometry x="30" y="405" width="60" height="20" as="geometry" />
        </mxCell>
        
        <!-- Frozen -->
        <mxCell id="gen-legend-frozen-box" value="" 
                style="rounded=0;whiteSpace=wrap;html=1;fillColor=#6FA8DC;strokeColor=#0066CC;strokeWidth=2;" 
                vertex="1" parent="1">
          <mxGeometry x="35" y="430" width="25" height="15" as="geometry" />
        </mxCell>
        <mxCell id="gen-legend-frozen-text" value="= frozen" 
                style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;" 
                vertex="1" parent="1">
          <mxGeometry x="65" y="427" width="80" height="20" as="geometry" />
        </mxCell>
        
        <!-- Trainable -->
        <mxCell id="gen-legend-train-box" value="" 
                style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFB366;strokeColor=#FF6600;strokeWidth=2;" 
                vertex="1" parent="1">
          <mxGeometry x="35" y="452" width="25" height="15" as="geometry" />
        </mxCell>
        <mxCell id="gen-legend-train-text" value="= trainable" 
                style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;" 
                vertex="1" parent="1">
          <mxGeometry x="65" y="449" width="80" height="20" as="geometry" />
        </mxCell>
        
        <!-- Zeros -->
        <mxCell id="gen-legend-zero-box" value="" 
                style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#999999;strokeWidth=2;" 
                vertex="1" parent="1">
          <mxGeometry x="35" y="474" width="25" height="15" as="geometry" />
        </mxCell>
        <mxCell id="gen-legend-zero-text" value="= zeros" 
                style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;" 
                vertex="1" parent="1">
          <mxGeometry x="65" y="471" width="80" height="20" as="geometry" />
        </mxCell>
        
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""

if __name__ == "__main__":
    with open("step1_general_diagram.xml", "w") as f:
        f.write(GENERAL_DIAGRAM)
    print("✓ Step 1: General SOARA diagram generated!")
