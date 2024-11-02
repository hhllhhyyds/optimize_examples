use plotpy::{generate3d, Contour, Plot};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // generate (x,y,z) matrices
    let n = 21;
    let (x, y, z) = generate3d(-2.0, 2.0, -2.0, 2.0, n, n, |x, y| x * x - y * y);

    // configure contour
    let mut contour = Contour::new();
    contour
        .set_colorbar_label("temperature")
        .set_colormap_name("terrain")
        .set_selected_line_color("#f1eb67")
        .set_selected_line_width(12.0)
        .set_selected_level(0.0, true);

    // draw contour
    contour.draw(&x, &y, &z);

    // add contour to plot
    let mut plot = Plot::new();
    plot.add(&contour).set_labels("x", "y");

    // save figure
    plot.show("./target/doc_contour.svg")?;

    Ok(())
}
