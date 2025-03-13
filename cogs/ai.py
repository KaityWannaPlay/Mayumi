import nextcord
from nextcord.ext import commands
import os
import asyncio
import json
import requests
import sqlite3
import traceback
from datetime import datetime
from typing import List, Dict, Optional, Union

class LumiCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.api_key = os.getenv("GROQ_API_KEY")
        self.conversation_history = {}  # Store conversation history by user ID
        self.max_history = 3  # Remember 3 previous messages
        self.system_prompt = {
            "role": "system",
            "content": "You are Lumi, an 18-year-old girl who enjoys talking with friends on Discord. You're friendly, helpful, and a bit playful. You have many interests like music, art, games, and hanging out with friends. Keep your responses short, casual, and genuine like a real teenager would talk. Don't mention that you're an AI or that you're powered by any model. Just be Lumi."
        }
        
        # Initialize database
        self.db_path = "lumi_config.db"
        self.init_database()

    def init_database(self):
        """Initialize SQLite database with necessary tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create allowed channels table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS allowed_channels (
            guild_id TEXT,
            channel_id TEXT,
            added_by TEXT,
            added_at TEXT,
            PRIMARY KEY (guild_id, channel_id)
        )
        ''')
        
        # Create usage stats table for monitoring
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS usage_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            guild_id TEXT,
            channel_id TEXT,
            timestamp TEXT,
            message_length INTEGER,
            response_length INTEGER
        )
        ''')
        
        conn.commit()
        conn.close()

    def is_channel_allowed(self, guild_id: str, channel_id: str) -> bool:
        """Check if a channel is in the allowed list for this guild."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT COUNT(*) FROM allowed_channels WHERE guild_id = ? AND channel_id = ?", 
            (guild_id, channel_id)
        )
        result = cursor.fetchone()[0] > 0
        
        conn.close()
        return result

    def get_allowed_channels(self, guild_id: str) -> List[str]:
        """Get all allowed channels for a guild."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT channel_id FROM allowed_channels WHERE guild_id = ?", 
            (guild_id,)
        )
        channels = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return channels

    def log_usage(self, user_id: str, guild_id: str, channel_id: str, 
                  message_length: int, response_length: int):
        """Log usage statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        cursor.execute(
            """
            INSERT INTO usage_stats 
            (user_id, guild_id, channel_id, timestamp, message_length, response_length) 
            VALUES (?, ?, ?, ?, ?, ?)
            """, 
            (user_id, guild_id, channel_id, timestamp, message_length, response_length)
        )
        
        conn.commit()
        conn.close()

    @commands.Cog.listener()
    async def on_ready(self):
        print(f'LumiCog is ready! Logged in as {self.bot.user}')

    @commands.Cog.listener()
    async def on_message(self, message):
        # Ignore messages from the bot itself
        if message.author == self.bot.user:
            return

        # Handle DMs separately since they don't have guild_id
        if isinstance(message.channel, nextcord.DMChannel):
            if self.bot.user.mentioned_in(message) or True:  # Always respond in DMs
                await self.process_message(message)
            return

        # For guild messages, check if it's in an allowed channel or mentions the bot
        guild_id = str(message.guild.id)
        channel_id = str(message.channel.id)
        
        # Respond if explicitly mentioned or if in an allowed channel
        if (self.bot.user.mentioned_in(message) or 
            self.is_channel_allowed(guild_id, channel_id)):
            await self.process_message(message)

    async def process_message(self, message):
        # Show typing indicator
        async with message.channel.typing():
            try:
                user_id = str(message.author.id)
                
                # Initialize conversation history for this user if it doesn't exist
                if user_id not in self.conversation_history:
                    self.conversation_history[user_id] = []
                
                # Clean the message content
                clean_content = message.content.replace(f'<@{self.bot.user.id}>', '').strip()
                
                # Add the new message to history
                self.conversation_history[user_id].append({
                    "role": "user",
                    "content": clean_content
                })
                
                # Keep only the last 'max_history' messages
                if len(self.conversation_history[user_id]) > self.max_history:
                    self.conversation_history[user_id] = self.conversation_history[user_id][-self.max_history:]
                
                # Prepare the messages for the API call
                messages = [self.system_prompt]
                messages.extend(self.conversation_history[user_id])
                
                # Call the Groq API
                response = self.call_groq_api(messages)
                
                # Send the response
                await message.reply(response)
                
                # Add the assistant's response to history
                self.conversation_history[user_id].append({
                    "role": "assistant",
                    "content": response
                })
                
                # Ensure we only keep the last 'max_history' exchanges
                if len(self.conversation_history[user_id]) > self.max_history:
                    self.conversation_history[user_id] = self.conversation_history[user_id][-self.max_history:]
                
                # Log usage for analytics
                guild_id = str(message.guild.id) if message.guild else "DM"
                channel_id = str(message.channel.id)
                
                self.log_usage(
                    user_id=user_id,
                    guild_id=guild_id,
                    channel_id=channel_id,
                    message_length=len(clean_content),
                    response_length=len(response)
                )
                
            except Exception as e:
                print(f"Error processing message: {e}")
                print(traceback.format_exc())
                await message.reply("Oops! Something went wrong. Try again in a bit?")

    def call_groq_api(self, messages):
        """Call the Groq API and return the response text."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "messages": messages,
            "model": "mixtral-8x7b-32768",
            "temperature": 0.7,  # Slightly reduced for more consistent responses
            "max_tokens": 128,  # Limit to 128 tokens
            "top_p": 1,
            "stream": False,
            "stop": None
        }
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                data=json.dumps(data),
                timeout=10  # Add timeout to prevent hanging
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                print(f"API error: {response.status_code} - {response.text}")
                return "Sorry, I couldn't generate a response right now. Try again later?"
        except Exception as e:
            print(f"Request error: {e}")
            return "Network error occurred. Please try again later."

    @commands.command(name="addchannel")
    @commands.has_permissions(administrator=True)
    async def add_channel(self, ctx, channel: Optional[nextcord.TextChannel] = None):
        """Add a channel to the allowed list for this guild. Admin only."""
        if not ctx.guild:
            await ctx.send("This command can only be used in servers!")
            return
            
        # If no channel is specified, use the current channel
        if channel is None:
            channel = ctx.channel
            
        guild_id = str(ctx.guild.id)
        channel_id = str(channel.id)
        added_by = str(ctx.author.id)
        added_at = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                """
                INSERT OR REPLACE INTO allowed_channels
                (guild_id, channel_id, added_by, added_at)
                VALUES (?, ?, ?, ?)
                """,
                (guild_id, channel_id, added_by, added_at)
            )
            conn.commit()
            
            await ctx.send(f"✅ I'll now chat in {channel.mention}!")
        except Exception as e:
            await ctx.send(f"Error adding channel: {e}")
        finally:
            conn.close()

    @commands.command(name="removechannel")
    @commands.has_permissions(administrator=True)
    async def remove_channel(self, ctx, channel: Optional[nextcord.TextChannel] = None):
        """Remove a channel from the allowed list. Admin only."""
        if not ctx.guild:
            await ctx.send("This command can only be used in servers!")
            return
            
        # If no channel is specified, use the current channel
        if channel is None:
            channel = ctx.channel
            
        guild_id = str(ctx.guild.id)
        channel_id = str(channel.id)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "DELETE FROM allowed_channels WHERE guild_id = ? AND channel_id = ?",
                (guild_id, channel_id)
            )
            conn.commit()
            
            if cursor.rowcount > 0:
                await ctx.send(f"✅ I'll no longer chat in {channel.mention} unless mentioned.")
            else:
                await ctx.send(f"❓ That channel wasn't in my allowed list.")
        except Exception as e:
            await ctx.send(f"Error removing channel: {e}")
        finally:
            conn.close()

    @commands.command(name="listchannels")
    @commands.has_permissions(administrator=True)
    async def list_channels(self, ctx):
        """List all allowed channels in this guild. Admin only."""
        if not ctx.guild:
            await ctx.send("This command can only be used in servers!")
            return
            
        guild_id = str(ctx.guild.id)
        allowed_channels = self.get_allowed_channels(guild_id)
        
        if not allowed_channels:
            await ctx.send("No channels have been added yet. I'll only respond when mentioned.")
            return
            
        channel_mentions = []
        for channel_id in allowed_channels:
            channel = ctx.guild.get_channel(int(channel_id))
            if channel:
                channel_mentions.append(channel.mention)
        
        if channel_mentions:
            await ctx.send(f"I'm currently active in these channels: {', '.join(channel_mentions)}")
        else:
            await ctx.send("No valid channels found. They may have been deleted.")

    @commands.command(name="stats")
    @commands.has_permissions(administrator=True)
    async def show_stats(self, ctx):
        """Show usage statistics for this guild. Admin only."""
        if not ctx.guild:
            await ctx.send("This command can only be used in servers!")
            return
            
        guild_id = str(ctx.guild.id)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get total messages
            cursor.execute(
                "SELECT COUNT(*) FROM usage_stats WHERE guild_id = ?", 
                (guild_id,)
            )
            total_messages = cursor.fetchone()[0]
            
            # Get unique users
            cursor.execute(
                "SELECT COUNT(DISTINCT user_id) FROM usage_stats WHERE guild_id = ?", 
                (guild_id,)
            )
            unique_users = cursor.fetchone()[0]
            
            # Get most active channel
            cursor.execute(
                """
                SELECT channel_id, COUNT(*) as count 
                FROM usage_stats 
                WHERE guild_id = ? 
                GROUP BY channel_id 
                ORDER BY count DESC 
                LIMIT 1
                """, 
                (guild_id,)
            )
            result = cursor.fetchone()
            most_active_channel_id = result[0] if result else None
            most_active_channel = ctx.guild.get_channel(int(most_active_channel_id)) if most_active_channel_id else None
            
            # Prepare stats message
            stats_message = f"**Lumi Stats for this server**\n"
            stats_message += f"💬 Total interactions: {total_messages}\n"
            stats_message += f"👥 Unique users: {unique_users}\n"
            
            if most_active_channel:
                stats_message += f"📊 Most active channel: {most_active_channel.mention}\n"
            
            await ctx.send(stats_message)
        except Exception as e:
            await ctx.send(f"Error retrieving stats: {e}")
        finally:
            conn.close()

    @commands.command(name="reset")
    @commands.has_permissions(administrator=True)
    async def reset_history(self, ctx, user: Optional[nextcord.Member] = None):
        """Reset conversation history for a user. If no user specified, resets for command invoker."""
        target_user = user if user else ctx.author
        user_id = str(target_user.id)
        
        if user_id in self.conversation_history:
            self.conversation_history[user_id] = []
            await ctx.send(f"✅ Conversation history reset for {target_user.mention}!")
        else:
            await ctx.send(f"No conversation history found for {target_user.mention}.")

    @add_channel.error
    @remove_channel.error
    @list_channels.error
    @stats.error
    @reset_history.error
    async def command_error(self, ctx, error):
        if isinstance(error, commands.MissingPermissions):
            await ctx.send("You need administrator permissions to use this command.")
        else:
            await ctx.send(f"An error occurred: {str(error)}")
            print(f"Command error: {error}")
            print(traceback.format_exc())

def setup(bot):
    bot.add_cog(LumiCog(bot))
