using System;
using System.IO;
using System.Collections;
using System.Linq;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;

namespace GateWay
{
    class Server //server to users
    {
        public List<string> filelist = null;
        private Client file_client = null;
        public Server(string address,int port ,Client client)
        {
            this.address = address;
            this.port = port;
            this.file_client = client;
            this.filelist = file_client.Get_filelist();
        }
        private  TcpListener listener { get; set; }
        private string address;
        private  int port;
        
        private  bool accept { get; set; } = false;
        public  void StartServer()
        {
            IPAddress address = IPAddress.Parse("127.0.0.1");
            listener = new TcpListener(address, port);
            listener.Start();
            accept = true;
            Console.WriteLine($"Server started. Listening to TCP clients at 127.0.0.1:{port}");
        }
        public  void Listen()
        {
            if (listener != null && accept)
            {
                while (true)
                {
                    Console.WriteLine("Waiting for client...");
                    var clientTask = listener.AcceptTcpClientAsync();
                    if (clientTask.Result != null)
                    {
                        while (true)
                        {
                            Console.WriteLine("Client connected. Waiting for data.");
                            var client = clientTask.Result;
                            string message = "";
                            byte[] data = Encoding.UTF8.GetBytes("send_request");
                            client.GetStream().Write(data, 0, data.Length);
                            byte[] buffer = new byte[1024];
                            client.GetStream().Read(buffer, 0, buffer.Length);
                            message = Encoding.UTF8.GetString(buffer).TrimEnd('\0');
                            Console.WriteLine(message);
                            if (message.StartsWith("list"))
                            {
                                foreach (var file in filelist)
                                {
                                    data = Encoding.UTF8.GetBytes(file);
                                    client.GetStream().Write(data, 0, data.Length);
                                    buffer = new byte[1024];
                                    client.GetStream().Read(buffer, 0, buffer.Length);
                                }
                                data = Encoding.UTF8.GetBytes("finished");
                                client.GetStream().Write(data, 0, data.Length);
                                continue;
                            }
                            if (message.StartsWith("quit"))
                            {
                                Console.WriteLine("Closing connection.");
                                client.GetStream().Dispose();
                                break;
                            }
                            foreach (var item in filelist)
                            {
                                if (message.StartsWith(item))
                                {
                                    var result=file_client.Get_file(item).TrimEnd('\0');
                                    Console.WriteLine(result);
                                    data = Encoding.UTF8.GetBytes(result);
                                    client.GetStream().Write(data, 0, data.Length);
                                }
                            }
                        }
                        
                    }

                }
            }
        }
        public void showAll()
        {
            foreach (var item in filelist)
            {
                Console.WriteLine(item);
            }
        }
    }
    class Client //client to sub_server
    {
        private TcpClient client = null;
        private IPEndPoint iep = null;
        private string address = null;
        private int port;
        public Client(string address,int port)
        {
            this.address = address;
            this.port = port;
            this.client = new TcpClient();
            this.iep = new IPEndPoint(IPAddress.Parse(this.address), this.port);
            this.client.Connect(iep);
        }
        public List<string> Get_filelist()
        {
            List<string> result = new List<string>();
            byte[] buffer = new byte[1024];
            client.GetStream().Read(buffer, 0, buffer.Length);
            byte[] data = Encoding.UTF8.GetBytes("getfilelist");
            client.GetStream().Write(data, 0, data.Length);
            while (true)
            {
                buffer = new byte[1024];
                client.GetStream().Read(buffer, 0, buffer.Length);
                string message = Encoding.UTF8.GetString(buffer).TrimEnd('\0');
   
                if (message.StartsWith("finished"))
                {
                    client.GetStream().Dispose();
                    break;
                }
                else
                {
                    //Console.WriteLine(message);
                    result.Add(message);
                    data = Encoding.UTF8.GetBytes("next");
                    client.GetStream().Write(data, 0, data.Length);
                }
            }
            return result;
        }
        public string Get_file(string filename)
        {
            this.client = new TcpClient();
            this.iep = new IPEndPoint(IPAddress.Parse(this.address), this.port);
            this.client.Connect(iep);
            byte[] buffer = new byte[1024];
            client.GetStream().Read(buffer, 0, buffer.Length);
            byte[] data = Encoding.UTF8.GetBytes(filename);
            client.GetStream().Write(data, 0, data.Length);
            buffer = new byte[1024];
            client.GetStream().Read(buffer, 0, buffer.Length);
            string message = Encoding.UTF8.GetString(buffer).TrimEnd('\0');
            data = Encoding.UTF8.GetBytes("OK");
            client.GetStream().Write(data, 0, data.Length);
            Console.WriteLine("got:"+message);
            return message;
        }
    }
    class Program
    {
        static void Main(string[] args)
        {
            Client file_client = new Client("127.0.0.1", 1080);
            Server gateway = new Server("127.0.0.1", 80,file_client);

            Console.WriteLine("filelist got");
            gateway.StartServer();
            gateway.Listen();
        }
    }
}
