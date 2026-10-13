fn main() {
    std::process::exit(ghee::cli::main(std::env::args().skip(1).collect()));
}
