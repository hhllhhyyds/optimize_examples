use plotpy::{generate3d, Contour, Plot};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // generate (x,y,z) matrices
    let n = 51;
    let (x, y, z) = generate3d(-2.0, 3.0, -2.0, 3.0, n, n, |x, y| {
        (1.0 - x).powi(2) + 5.0 * (y - x.powi(2)).powi(2)
    });

    // configure contour
    let mut contour = Contour::new();
    contour
        .set_colorbar_label("z")
        .set_colormap_name("terrain")
        .set_selected_line_color("#f1eb67")
        .set_selected_line_width(0.1)
        .set_levels(
            &(0..10)
                .map(|i| 0.01 + (i as f64).powf(2.5))
                .collect::<Vec<_>>(),
        )
        .set_no_labels(true);

    // draw contour
    contour.draw(&x, &y, &z);

    // add contour to plot
    let mut plot = Plot::new();
    plot.add(&contour).set_labels("x", "y");

    // save figure
    plot.show("./target/doc_contour.svg")?;

    Ok(())
}
