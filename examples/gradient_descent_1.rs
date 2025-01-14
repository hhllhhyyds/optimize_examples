use plotpy::{generate3d, Contour, Curve, Plot};

use optimize_examples::{bracketing, test_math_funcs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = 1.;
    let b = 5.;
    let rosenbrock_a = |x: &[f64; 2]| -> f64 { test_math_funcs::rosenbrock(x, a, b) };
    let rosenbrock_grad_a =
        |x: &[f64; 2]| -> [f64; 2] { test_math_funcs::rosenbrock_grad(x, a, b) };

    let n = 81;
    let (x, y, z) = generate3d(-3.0, 3.0, -3.0, 3.0, n, n, |x, y| rosenbrock_a(&[x, y]));

    // configure contour
    let mut contour = Contour::new();
    contour
        .set_colorbar_label("z")
        .set_colormap_name("terrain")
        .set_selected_line_color("#f1eb67")
        .set_selected_line_width(0.01)
        .set_levels(
            &(0..10)
                .map(|i| 0.01 + (i as f64).powf(2.5))
                .collect::<Vec<_>>(),
        )
        .set_no_labels(true);

    // draw contour
    contour.draw(&x, &y, &z);

    let iter_count = 100;
    let mut start_point = [-2., -1.];
    let mut iter_points = vec![];

    for _ in 0..iter_count {
        iter_points.push(start_point);

        let grad = rosenbrock_grad_a(&start_point);
        let grad_norm = (grad[0].powi(2) + grad[1].powi(2)).sqrt();
        let d = grad.map(|x| -x / grad_norm);

        let f = |alpha| {
            let x = start_point[0] + alpha * d[0];
            let y = start_point[1] + alpha * d[1];
            rosenbrock_a(&[x, y])
        };

        let bracket = bracketing::bracket_minimum(&f, 0., 1e-3, 2., 100).unwrap();
        let bracket = bracketing::golden_section_search(&f, bracket, 30);
        let learning_rate = (bracket.0 + bracket.1) / 2.;

        start_point[0] += learning_rate * d[0];
        start_point[1] += learning_rate * d[1];
    }

    let mut curve = Curve::new();
    curve.set_line_color("#f1eb67");
    curve.set_line_width(0.5);
    curve.points_begin();
    iter_points.into_iter().for_each(|p| {
        curve.points_add(p[0], p[1]);
    });
    curve.points_end();

    // add contour to plot
    let mut plot = Plot::new();
    plot.add(&contour).set_labels("x", "y");
    plot.add(&curve);

    // save figure
    plot.show("./target/gradient_descent.svg")?;

    Ok(())
}
