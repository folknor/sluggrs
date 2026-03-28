use iced::widget::{center, column, text};
use iced::Element;

fn main() -> iced::Result {
    iced::application(App::new, App::update, App::view)
        .window_size((800.0, 600.0))
        .default_font(iced::Font::DEFAULT)
        .run()
}

struct App;

#[derive(Debug, Clone)]
enum Message {}

impl App {
    fn new() -> (Self, iced::Task<Message>) {
        (App, iced::Task::none())
    }

    fn update(&mut self, _message: Message) -> iced::Task<Message> {
        iced::Task::none()
    }

    fn view(&self) -> Element<'_, Message> {
        center(
            column![
                text("Hello 👋 World 🎉 Emoji Test 😀🦊🐕").size(32),
                text("The quick brown fox 🦊 jumps over the lazy dog 🐕").size(24),
                text("Sluggrs + Raster Fallback 🚀✨🎨").size(20),
                text("Pure vector text: no emoji here").size(16),
                text("Mixed: abc 🔥 def 💧 ghi ⚡ jkl").size(14),
            ]
            .spacing(20),
        )
        .into()
    }
}
