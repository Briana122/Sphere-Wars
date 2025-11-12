import pygame

# PlayerUI Class:
# This Class displays information about an individual player in the game.
# To use this class, you will need to create an instance of PlayerUI for each player.
# If you wish, you may customize the font, colors, margin, background color, and text color to your liking.
# Otherwise, call it as is and it will use the defaults.
# Afterwards, you can call any of the draw funtions that you want, depending on what information you need.
class PlayerUI:
    def __init__(self, player_index: int, font_name=None, font_size: int = 18, margin: int = 8,
                 bg_color=(0, 0, 0, 160), text_color=(255, 255, 255), column_spacing: int = 140):
        pygame.font.init()
        self.player_index = int(player_index)
        self.font = pygame.font.SysFont(font_name, font_size)
        self.margin = margin
        self.bg_color = bg_color
        self.text_color = text_color
        self.column_spacing = column_spacing
    
    def compute_tile_count(self, game):
        return game.player_objs[self.player_index].get_num_tiles_owned()

    def compute_resource_count(self, game):
        return game.player_objs[self.player_index].get_current_resources(game)
    
    def compute_piece_count(self, game):
        return game.player_objs[self.player_index].get_num_pieces_owned()
    
    # This is what the lines will look like when drawn on the screen.
    def format_lines(self, game):
        tiles = self.compute_tile_count(game)
        resources = self.compute_resource_count(game)
        pieces = self.compute_piece_count(game)
        return [
            f"Player {self.player_index + 1}:",
            f"  Tiles: {tiles}",
            f"  Resources: {resources}",
            f"  Pieces: {pieces}",
        ]
    
    # Draws the player stats on the given surface at the specified base position.
    def draw_player_stats(self, surface: pygame.Surface, game, base_pos=(10, 10), column_spacing: int = None):
        # If column spacing is provided, use it.
        # else use what you already have stored.
        if column_spacing is not None:
            spacing = column_spacing
        else:
            spacing = self.column_spacing

        # Based on the player index, we will offset the x position.
        x = base_pos[0] + self.player_index * spacing
        y = base_pos[1]

        # Prepare the text surfaces to be drawn.
        lines = self.format_lines(game)
        text_surfs = []

        # Render each line of text.
        for line in lines:
            text_surfs.append(self.font.render(line, True, self.text_color))

        # Calculate the width and height of the background rectangle using the text surfaces.
        width = max(s.get_width() for s in text_surfs) + self.margin * 2
        height = sum(s.get_height() for s in text_surfs) + self.margin * (len(text_surfs) + 1)

        # Draw the background rectangle and fill it with the background color.
        bg = pygame.Surface((width, height), pygame.SRCALPHA)
        bg.fill(self.bg_color)

        # Draw the background onto the main surface.
        surface.blit(bg, (x, y))

        # Start drawing text just below the top margin.
        yy = y + self.margin

        # Draw each line of text onto the main surface.
        for s in text_surfs:
            surface.blit(s, (x + self.margin, yy))
            yy += s.get_height() + self.margin


