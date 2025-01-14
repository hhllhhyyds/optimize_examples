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

fn draw_contour(contour: &mut Contour, f: impl Fn(&DVec2) -> f64) {
    let n = 81;
    let (x, y, z) = generate3d(-1.5, 1.5, -2.0, 4.0, n, n, |x, y| f(&DVec2::new(x, y)));

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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = 1.;
    let b = 100.;
    let rosenbrock_a = |x: &DVec2| -> f64 { test_math_funcs::rosenbrock(&x.to_array(), a, b) };
    let rosenbrock_grad_a =
        |x: &DVec2| -> DVec2 { test_math_funcs::rosenbrock_grad(&x.to_array(), a, b).into() };

    let mut contour = Contour::new();
    draw_contour(&mut contour, rosenbrock_a);

    let iter_count = 30;

    let mut curve1 = Curve::new();
    {
        let mut start_point = DVec2::new(-1.4, 0.9);
        let mut iter_points = vec![];

        for _ in 0..iter_count {
            iter_points.push(start_point);

            let d = -rosenbrock_grad_a(&start_point).normalize();
            let f = |alpha| rosenbrock_a(&(start_point + alpha * d));

            let bracket = bracketing::bracket_minimum(&f, 0., 1e-8, 2., 100).unwrap();
            let bracket = bracketing::golden_section_search(&f, bracket, 3);
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
        let mut start_point = DVec2::new(-1.4, 1.);
        let mut iter_points = vec![start_point];

        let mut g = rosenbrock_grad_a(&start_point);
        let mut d = -g;

        for _ in 0..iter_count {
            let f = |alpha| rosenbrock_a(&(start_point + alpha * d));

            let bracket = bracketing::bracket_minimum(&f, 0., 1e-8, 2., 100).unwrap();
            let bracket = bracketing::golden_section_search(&f, bracket, 3);
            let learning_rate = (bracket.0 + bracket.1) / 2.;

            start_point += learning_rate * d;
            iter_points.push(start_point);

            let g_new = rosenbrock_grad_a(&start_point);
            let beta = ((g_new - g).dot(g_new) / g.length_squared()).max(0.);
            d = -g_new + beta * d;
            g = g_new;
        }

        curve2.set_line_color("white");
        curve2.set_line_width(0.5);
        curve2.set_label("conjugate gradient descent");
        add_points_to_curve(&mut curve2, &iter_points);
    }

    let mut curve3 = Curve::new();
    {
        let mut start_point = DVec2::new(-1.4, 1.1);
        let mut iter_points = vec![];
        let mut v = DVec2::ZERO;

        for _ in 0..iter_count {
            iter_points.push(start_point);

            let d = -rosenbrock_grad_a(&start_point).normalize();
            let f = |alpha| rosenbrock_a(&(start_point + alpha * d));

            let bracket = bracketing::bracket_minimum(&f, 0., 1e-8, 2., 100).unwrap();
            let bracket = bracketing::golden_section_search(&f, bracket, 3);
            let learning_rate = (bracket.0 + bracket.1) / 2.;

            v = 0.7 * v + learning_rate * d;

            start_point += v;
        }

        curve3.set_line_color("pink");
        curve3.set_line_width(0.5);
        curve3.set_label("momentum gradient descent");
        add_points_to_curve(&mut curve3, &iter_points);
    }

    let mut plot = Plot::new();
    plot.add(&contour).set_labels("x", "y");
    plot.add(&curve1);
    plot.add(&curve2);
    plot.add(&curve3);
    plot.legend();

    plot.show("./target/gradient_descent_2.svg")?;

    Ok(())
}
