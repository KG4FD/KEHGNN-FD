import torch
import torch.nn.functional as F
import networkx as nx
import pickle
import random
from tqdm import tqdm
import numpy as np
from torch_geometric.nn import GATConv, Linear, to_hetero
from torch_geometric.data import HeteroData

def content_nodes(g):
    text_nodes, entity_nodes, topic_nodes = set(), set(), set()
    for i in g.nodes():
        if i.isdigit():
            text_nodes.add(i)
        elif i[:6] == "topic_":
            topic_nodes.add(i)
        else:
            entity_nodes.add(i)
    print("# news_nodes: {}       # entity_nodes: {}       # topic_nodes: {}".format(
        len(text_nodes), len(entity_nodes), len(topic_nodes)
    ))

    return text_nodes, entity_nodes, topic_nodes

def load_dataset(dataset):
    root_dir = "../data"
    if dataset == "FakeNewsNet":
        dataset_name = "FakeNewsNet/completed_data/"
    if dataset == "ISOT":
        dataset_name = "ISOT/completed_data/"
    if dataset == "LIAR_PANTS":
        dataset_name = "LIAR_PANTS/completed_data/"
    if dataset == "pan2020":
        dataset_name = "pan2020/completed_data/"
    
    dataset_dir = root_dir + dataset_name
    dataset_name = dataset_name.split("/")[0]
    print("Dataset: " + dataset_name)
    
    News_Dict = dict()
    Topic_Dict = dict()
    Entity_Dict = dict()

    # read the news attribute file
    entity_list = []
    entity_attr = []
    with open(dataset_dir + dataset_name + '.content.entity', 'r', encoding='utf8') as f:
        for line in f:
            entity_id, entity_attribute = line.strip('\n').split('\t')[0], line.strip('\n').split('\t')[1:-2]
            entity_attribute = np.array(entity_attribute, dtype=np.float64)
            entity_attribute = torch.Tensor(entity_attribute)
            entity_attr.append(entity_attribute)
            entity_list.append(entity_id)
    entity_attr = torch.stack(entity_attr)

    for i, entity_id in enumerate(entity_list):
        Entity_Dict.update({
            entity_id : i
        })

    topic_list = []
    topic_attr = []
    with open(dataset_dir + dataset_name + '.content.topic', 'r', encoding='utf8') as f:
        for line in f:
            topic_id, topic_attribute = line.strip('\n').split('\t')[0], line.strip('\n').split('\t')[1:-1]
            topic_attribute = np.array(topic_attribute, dtype=np.float64)
            topic_attribute = torch.Tensor(topic_attribute)
            topic_attr.append(topic_attribute)
            topic_list.append(topic_id)
    topic_attr = torch.stack(topic_attr)

    for i, topic_id in enumerate(topic_list):
        Topic_Dict.update({
            topic_id : i
        })

    news_list = []
    news_attr = []
    with open(dataset_dir + dataset_name + '.content.text', 'r', encoding='utf8') as f:
        for line in f:
            news_id, news_attribute = line.strip('\n').split('\t')[0], line.strip('\n').split('\t')[1:]
            news_attribute = np.array(news_attribute, dtype=np.float64)
            news_attribute = torch.Tensor(news_attribute)
            news_attr.append(news_attribute)
            news_list.append(news_id)
    news_attr = torch.stack(news_attr)

    for i, news_id in enumerate(news_list):
        News_Dict.update({
            int(news_id) : i 
        })
    
    # Read the raw graph from .pkl
    g = nx.Graph()
    graph_path = dataset_dir + "model_network_handled.pkl"
    with open(graph_path, 'rb') as f:
        g= pickle.load(f)
    
    news_nodes, entity_nodes, topic_nodes = content_nodes(g)
    mapindex = {}
    with open(dataset_dir+ 'mapindex.txt', 'r', encoding='utf8') as f:
        for line in f:
            old_index, index = line.strip('\n').split('\t')
            mapindex.update({
                old_index: int(index)
            })
    
    n_h_e = []
    e_i_n = []

    n_o_t = []
    t_i_n = []

    e_s_e = []

    for edge in g.edges:
        s_node = edge[0]
        d_node = edge[1]

        if s_node.isdigit() and d_node[:6] != "topic_":
            s_node = torch.Tensor([int(News_Dict[mapindex[s_node]])])
            d_node = torch.Tensor([int(Entity_Dict[str(mapindex[str(d_node)])])])
            t = torch.cat((s_node, d_node), 0 )
            t_ = torch.cat((d_node, s_node), 0 )
            n_h_e.append(t)
            e_i_n.append(t_)
            
        elif d_node.isdigit() and s_node[:6] != "topic_":
            s_node = torch.Tensor([int(Entity_Dict[str(mapindex[str(s_node)])])])
            d_node = torch.Tensor([int(News_Dict[mapindex[d_node]])])
            t = torch.cat((s_node, d_node),0)
            t_ = torch.cat((d_node, s_node), 0 )
            n_h_e.append(t_)
            e_i_n.append(t)

        # Get news - on - topic edgelist
        elif s_node.isdigit() and d_node[:6] == "topic_":
            s_node = torch.Tensor([int(News_Dict[mapindex[s_node]])])
            d_node = torch.Tensor([int(Topic_Dict[str(mapindex[str(d_node)])])])
            t = torch.cat((s_node, d_node),0)
            t_ = torch.cat((d_node, s_node),0)
            n_o_t.append(t)
            t_i_n.append(t_)
        
        elif d_node.isdigit() and s_node[:6] == "topic_":
            s_node = torch.Tensor([int(Topic_Dict[str(mapindex[str(s_node)])])])
            d_node = torch.Tensor([int(News_Dict[mapindex[d_node]])])
            t = torch.cat((s_node, d_node),0)
            t_ = torch.cat((d_node, s_node),0)
            n_o_t.append(t_)
            t_i_n.append(t)            

        # Get entity - similar - entity edgelist
        elif not(s_node.isdigit()) and s_node[:6] != "topic_"  and d_node[:6] != "topic_" and not(d_node.isdigit()):
            s_node = torch.Tensor([int(Entity_Dict[str(mapindex[str(s_node)])])])
            d_node = torch.Tensor([int(Entity_Dict[str(mapindex[str(d_node)])])])
            t = torch.cat((s_node, d_node),0)
            e_s_e.append(t)

    news_label = []
    for i in g.nodes():
        if i.isdigit():
            if g.nodes[i]['type'] == '0':
                lb = torch.LongTensor([0])
            else:
                lb = torch.LongTensor([1])
            news_label.append(lb)
    n_h_e1 = torch.stack(n_h_e).to(torch.long)
    e_i_n1 = torch.stack(e_i_n).to(torch.long)

    n_o_t1 = torch.stack(n_o_t).to(torch.long)
    t_i_n1 = torch.stack(t_i_n).to(torch.long)

    e_s_e1 = torch.stack(e_s_e).to(torch.long)

    news_label1 = torch.stack(news_label)

    with open(dataset_dir + "ent_attr_kg_transe.pkl", 'rb') as f:
        kg= pickle.load(f)

    entity_kg_attr = torch.zeros((len(Entity_Dict),512))

    for ent in kg:
        idx = int(Entity_Dict[str(mapindex[str(ent)])])
        entity_kg_attr[idx] = torch.from_numpy(kg[ent])
    
    hgraph = HeteroData()

    hgraph['news'].x = news_attr
    hgraph['news'].y = news_label1.view(-1)
    hgraph['entities'].x = entity_attr
    hgraph['topic'].x = topic_attr
    hgraph['kg_entities'].x = entity_kg_attr

    hgraph['news', 'has', 'entities'].edge_index = (n_h_e1.t().contiguous())
    hgraph['entities', 'in', 'news'].edge_index = (e_i_n1.t().contiguous())

    hgraph['news', 'on', 'topic'].edge_index = (n_o_t1.t().contiguous())
    hgraph['topic', 'in', 'news'].edge_index = (t_i_n1.t().contiguous())

    hgraph['entities', 'similar', 'entities'].edge_index = (e_s_e1.t().contiguous())

    hgraph['news', 'has', 'kg_entities'].edge_index = (n_h_e1.t().contiguous())
    hgraph['kg_entities', 'in', 'news'].edge_index = (e_i_n1.t().contiguous())

    return hgraph

def shuffle_data(data, args):
    # Shuffle the data before training.
    train_ratio = args.train_ratio
    mask = list(range(data['news'].y.shape[0]))
    random.shuffle(mask)
    train_mask = mask[:int(train_ratio*len(mask))]
    test_mask = mask[int(train_ratio*len(mask)):]
    data['news'].train_mask = torch.LongTensor(train_mask)
    data['news'].test_mask = torch.LongTensor(test_mask)

    return data
