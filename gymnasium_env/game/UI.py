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
        count = 0
        # Check all tiles to see if they are owned by this playr.
        # Note: The way I do it right now checks every single tile in the game.
        # This can be really bad performance wise if there's a lot of tiles.
        # I think we should make the player keep track of what tiles they own, unless it's already there and I'm dumb.
        for tile in game.tiles.values():
            if tile.owner == self.player_index:
                count += 1
        return count

    def compute_resource_count(self, game):
        # Note: Shouldn't the player have a resources attribute instead of me having to access the game's resources?
        # This was a bit confusing to figure out unless again I'm missing something again.
        return game.resources.get(self.player_index, 0)
    
    def compute_piece_count(self, game):
        # Same here, shouldn't the player have a pieces attribute?
        count = 0
        for (aid, pid), piece in game.pieces.items():
            if aid == self.player_index:
                count += 1
        return count
    
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
        bg = pygame.Surface((width, height))
        bg.fill(self.bg_color)

        # Draw the background onto the main surface.
        surface.blit(bg, (x, y))

        # Start drawing text just below the top margin.
        yy = y + self.margin

        # Draw each line of text onto the main surface.
        for s in text_surfs:
            surface.blit(s, (x + self.margin, yy))
            yy += s.get_height() + self.margin


