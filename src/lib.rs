mod allocator;
mod translator;

#[cfg(test)]
mod tests {
    use crate::allocator::Allocator;

    use super::*;

    #[test]
    fn it_works() {
        let mut allocator = Allocator::new(4096, 0.5, 32);
        let free_block = allocator.alloc_cache();

        println!("Got block at virtual ID {}", free_block.0);
    }
}
