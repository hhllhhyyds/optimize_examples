use plotpy::{generate3d, Contour, Curve, Plot};

use optimize_examples::{bracketing, test_math_funcs};

use glam::DVec2;

fn add_points_to_curve(curve: &mut Curve, points: &[DVec2]) {
    curve.points_begin();
    points.iter().for_each(|p| {
        curve.points_add(p.x, p.y);
    });
    curve.points_end();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = 1.;
    let b = 5.;
    let rosenbrock_a = |x: &DVec2| -> f64 { test_math_funcs::rosenbrock(&x.to_array(), a, b) };
    let rosenbrock_grad_a =
        |x: &DVec2| -> DVec2 { test_math_funcs::rosenbrock_grad(&x.to_array(), a, b).into() };

    let n = 81;
    let (x, y, z) = generate3d(-3.0, 3.0, -3.0, 3.0, n, n, |x, y| {
        rosenbrock_a(&DVec2::new(x, y))
    });

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
    contour.draw(&x, &y, &z);

    let mut curve1 = Curve::new();
    {
        let iter_count = 70;
        let mut start_point = DVec2::new(-2., -1.);
        let mut iter_points = vec![];

        for _ in 0..iter_count {
            iter_points.push(start_point);

            let d = -rosenbrock_grad_a(&start_point).normalize();
            let f = |alpha| rosenbrock_a(&(start_point + alpha * d));

            let bracket = bracketing::bracket_minimum(&f, 0., 1e-8, 2., 100).unwrap();
            let bracket = bracketing::golden_section_search(&f, bracket, 30);
            let learning_rate = (bracket.0 + bracket.1) / 2.;

            start_point += learning_rate * d;
        }

        curve1.set_line_color("red");
        curve1.set_line_width(0.5);
        curve1.set_label("gradient descent");
        add_points_to_curve(&mut curve1, &iter_points);
    }

    let mut curve2 = Curve::new();
    {
        let iter_count = 10;
        let mut start_point = DVec2::new(-2.1, -1.1);
        let mut iter_points = vec![start_point];

        let mut g = rosenbrock_grad_a(&start_point);
        let mut d = -g;

        for _ in 0..iter_count {
            let f = |alpha| rosenbrock_a(&(start_point + alpha * d));

            let bracket = bracketing::bracket_minimum(&f, 0., 1e-8, 2., 100).unwrap();
            let bracket = bracketing::golden_section_search(&f, bracket, 30);
            let learning_rate = (bracket.0 + bracket.1) / 2.;

            start_point += learning_rate * d;
            iter_points.push(start_point);

            let g_new = rosenbrock_grad_a(&start_point);
            let beta = ((g_new - g).dot(g_new) / g.length_squared()).max(0.);
            d = -g_new + beta * d;
            g = g_new;
        }

        curve2.set_line_color("pink");
        curve2.set_line_width(0.5);
        curve2.set_label("conjugate gradient descent");
        add_points_to_curve(&mut curve2, &iter_points);
    }

    let mut plot = Plot::new();
    plot.add(&contour).set_labels("x", "y");
    plot.add(&curve1);
    plot.add(&curve2);
    plot.legend();

    plot.show("./target/gradient_descent.svg")?;

    Ok(())
}
