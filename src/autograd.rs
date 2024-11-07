pub mod compute_graph;

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use compute_graph::{basic_fn::BasicFn, node::Node};

    use approx::assert_abs_diff_eq;
    use rand::Rng;

    use super::*;

    #[test]
    fn auto_computational_graph_0() {
        fn logistic2(z: Node) -> Node {
            let exp_fn = BasicFn::exp();
            let exp_node_fn = exp_fn.to_gen_node_fn();
            1.0 / (1.0 + exp_node_fn(&[Rc::new(-z).into()]))
        }

        let y = logistic2(Node::start(1.5));

        println!("\ny = {:?}", y);
        assert!((y.value() - 0.818).abs() < 1e-3);
    }

    #[test]
    fn auto_computational_graph_1() {
        fn somefunc(x: Node, y: Node) -> Node {
            let exp_fn = BasicFn::exp();
            let exp_node_fn = exp_fn.to_gen_node_fn();
            (x.clone() * y.clone() + exp_node_fn(&[x.into()]) * exp_node_fn(&[y.clone().into()]))
                / (4.0 * y)
        }

        let y = somefunc(Node::start(3.0), Node::start(4.0));

        println!("\ny = {:?}", y);
    }

    #[test]
    fn auto_computational_graph_2() {
        fn somefunc(x: Node, y: Node) -> Node {
            let max_fn = BasicFn::new(
                Rc::new(|inputs| {
                    debug_assert!(inputs.len() == 2);
                    inputs[0].as_ref().max(*inputs[1].as_ref())
                }),
                Rc::new(|inputs| {
                    debug_assert!(inputs.len() == 2);
                    if inputs[0].as_ref() > inputs[1].as_ref() {
                        vec![1.0, 0.0]
                    } else {
                        vec![0.0, 1.0]
                    }
                }),
            )
            .with_info("f64 max");
            let max_node_fn = max_fn.to_gen_node_fn();
            BasicFn::ln().to_gen_node_fn()(&[(x.clone() * y
                + max_node_fn(&[x.into(), 2.0.into()]))
            .into()])
        }

        let y = somefunc(Node::start(3.0), Node::start(2.0));

        println!("\ny = {:?}", y);
    }

    #[test]
    fn auto_grad_0() {
        let f_0 = |x: Node| {
            let sin_fn = BasicFn::new(
                Rc::new(|inputs| {
                    debug_assert!(inputs.len() == 1);
                    inputs[0].as_ref().sin()
                }),
                Rc::new(|inputs| {
                    debug_assert!(inputs.len() == 1);
                    vec![inputs[0].as_ref().cos()]
                }),
            )
            .with_info("f64 sin");
            let sin_node_fn = sin_fn.to_gen_node_fn();
            x.clone() * sin_node_fn(&[x.into()])
        };

        let f_1 = |x: f64| x * x.sin();
        let g_1 = |x: f64| x.sin() + x * x.cos();

        let n = 20;
        (0..n)
            .map(|i| (i as f64 / n as f64) * std::f64::consts::TAU)
            .for_each(|x| {
                let y_0 = Rc::new(f_0(Node::start(x)));
                let y_1 = f_1(x);
                let grads = Node::auto_grad(y_0.clone());
                let g_0 = grads.first().unwrap();
                let g_1 = g_1(x);

                assert!(y_0.value() == y_1);
                assert!(g_0.node().value() == x);
                assert!(g_0.grad() == g_1);
            });
    }

    #[test]
    fn auto_grad_1() {
        let f_0 = |a: Node, b: Node| {
            let max_2_fn = BasicFn::new(
                Rc::new(|inputs| {
                    debug_assert!(inputs.len() == 1);
                    inputs[0].as_ref().max(2.0)
                }),
                Rc::new(|inputs| {
                    debug_assert!(inputs.len() == 1);
                    if inputs[0].as_ref() > &2.0 {
                        vec![1.0]
                    } else {
                        vec![0.0]
                    }
                }),
            )
            .with_info("f64 max");
            let max_node_fn = max_2_fn.to_gen_node_fn();
            BasicFn::ln().to_gen_node_fn()(&[(a.clone() * b + max_node_fn(&[a.into()])).into()])
        };

        let y = Rc::new(f_0(Node::start(3.0), Node::start(2.0)));
        let grads = Node::auto_grad(y.clone());

        assert!(y.value() == 9.0_f64.ln());
        assert!(grads[0].grad() == 1.0 / 3.0,);
        assert!(grads[1].grad() == 1.0 / 3.0,);
    }

    #[test]
    fn auto_grad_2() {
        let f_0 = |x: Node, y: Node, z: Node| {
            let a0 = x.clone() + y.clone();
            let a0 = BasicFn::sin().to_gen_node_fn()(&[a0.into()]);
            let b0 = y.clone() * z.clone();
            let b0 = BasicFn::cos().to_gen_node_fn()(&[b0.into()]);
            let c0 = x.clone() * x.clone() - z.clone() * z.clone();
            let c0 = BasicFn::ln().to_gen_node_fn()(&[c0.into()]);
            let d = a0 + b0 * c0;
            BasicFn::exp().to_gen_node_fn()(&[d.into()])
        };
        let f_1 =
            |x: f64, y: f64, z: f64| ((x + y).sin() + (y * z).cos() * (x * x - z * z).ln()).exp();
        let g_1_x = |x: f64, y: f64, z: f64| {
            f_1(x, y, z) * ((x + y).cos() + (y * z).cos() / (x * x - z * z) * 2.0 * x)
        };
        let g_1_y = |x: f64, y: f64, z: f64| {
            f_1(x, y, z) * ((x + y).cos() - (y * z).sin() * z * (x * x - z * z).ln())
        };
        let g_1_z = |x: f64, y: f64, z: f64| {
            f_1(x, y, z)
                * (-(y * z).sin() * y * (x * x - z * z).ln()
                    - (y * z).cos() / (x * x - z * z) * 2.0 * z)
        };

        let mut rng = rand::thread_rng();

        for _ in 0..10 {
            let z = rng.gen_range(0.1..1.0);
            let x = z + rng.gen_range(0.1..1.0);
            let y = rng.gen_range(2.1..3.9);

            let x_node = Node::start(x);
            let y_node = Node::start(y);
            let z_node = Node::start(z);

            let y_0 = f_0(x_node, y_node, z_node);
            let y_1 = f_1(x, y, z);

            assert!(y_0.value() == y_1);

            let grads = Node::auto_grad(Rc::new(y_0));
            let grad_x = g_1_x(x, y, z);
            let grad_y = g_1_y(x, y, z);
            let grad_z = g_1_z(x, y, z);

            for g in &grads {
                assert!([x, y, z].contains(&g.node().value()));
                if g.node().value() == x {
                    assert_abs_diff_eq!(grad_x, g.grad(), epsilon = 100.0 * f64::EPSILON);
                }
                if g.node().value() == y {
                    assert_abs_diff_eq!(grad_y, g.grad(), epsilon = 100.0 * f64::EPSILON);
                }
                if g.node().value() == z {
                    assert_abs_diff_eq!(grad_z, g.grad(), epsilon = 100.0 * f64::EPSILON);
                }
            }
        }
    }
}
