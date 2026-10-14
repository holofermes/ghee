import ghee

scene = ghee.Scene(
    params=[ghee.Param("ink", "color", "orange")],
    passes=[
        ghee.Pass(
            "main",
            "void main() { gl_FragColor = vec4(ink * (1.0 - uv.y), 1.0); }",
            layers=[ghee.Layer("title", text="hello", text_shadow={})],
        )
    ],
)
screen = ghee.open(scene, width=128, height=64)
screen.frame()
screen.save("scene.png")
